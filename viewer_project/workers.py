import os
import time
import cv2
import json
from pathlib import Path
import numpy as np
import threading
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker

import torch
import torch.nn as nn
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import config
from config import *
from utils_vision import (
    _draw_yolo, _draw_rfdetr, _draw_dinov3, _render_pointcloud_depth
)
from pointcloud_export import (
    export_segmented_pointclouds,
    extract_segmentation_regions,
    render_depth_segmentation_overlay,
    render_pcl_normals_view,
)

# Soporte para ROS 2
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import Image as ROSImage
    from sensor_msgs.msg import PointCloud2
    from sensor_msgs.msg import PointField
    from cv_bridge import CvBridge
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False

# https://doc.qt.io/qtforpython-6/PySide6/QtCore/QThread.html
# https://docs.python.org/3/library/threading.html
class DINOv3Model:
    """Loads the trained DINOv3 semantic segmenter and runs predictions on BGR frames."""

    def __init__(self, checkpoint_path, device=DEVICE):
        self.path = checkpoint_path
        self.error = None
        self.ready = False

        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            state_dict = checkpoint['model_state_dict']
            model = DINOv3SegmentationModel()
            model.load_state_dict(state_dict, strict=True)
            self.model = model.to(device).eval()
            self.device = device
            self.ready = True
        except Exception as exc:
            self.error = str(exc)

    def predict(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
        image = image.astype('float32') / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self.model(tensor)
            channels = logits.softmax(dim=1)[0]
            # Match the channel-to-class mapping used by DINOv3/test.py.
            class_probabilities = torch.stack((
                channels[1], channels[3], channels[2], channels[4],
                channels[0] + channels[5],
            ))
        return class_probabilities.cpu().numpy()


class DINOv3SegmentationModel(nn.Module):
    def __init__(self, num_classes=6, model_name='dinov3_vitb16'):
        super().__init__()
        backbone_weights = os.environ.get('DINOV3_WEIGHTS')
        load_kwargs = (
            {'weights': backbone_weights}
            if backbone_weights else {'pretrained': False}
        )
        self.backbone = torch.hub.load(
            'facebookresearch/dinov3', model_name, **load_kwargs
        )
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        in_features = getattr(self.backbone, 'embed_dim', 768)
        self.segmentation_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, image):
        features = self.backbone.get_intermediate_layers(image, n=1)[0]
        logits = self.segmentation_head(features)
        batch_size = image.shape[0]
        grid_height = image.shape[-2] // self.backbone.patch_size
        grid_width = image.shape[-1] // self.backbone.patch_size
        logits = logits.transpose(1, 2).reshape(
            batch_size, -1, grid_height, grid_width
        )
        return nn.functional.interpolate(
            logits, size=image.shape[-2:], mode='bilinear', align_corners=False
        )


# Backward-compatible alias for older imports/tests.
DINOv3Preview = DINOv3Model


class ModelLoader(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(object, object, object, object)
    error    = pyqtSignal(str)

    def run(self):
        self.load_errors = {}
        y12 = y26 = rf = dino = None

        try:
            from ultralytics import YOLO
            for name, path in (("YOLO12n-seg", YOLO12_PATH), ("YOLO26n-seg", YOLO26_PATH)):
                try:
                    self.progress.emit(f"Loading {name}…")
                    model = YOLO(path)
                    if hasattr(model, 'to'):
                        model.to(DEVICE)
                    if name == "YOLO12n-seg": y12 = model
                    else: y26 = model
                except Exception as ex:
                    self.load_errors[name] = str(ex)
        except Exception as ex:
            self.load_errors["YOLO"] = str(ex)

        try:
            from rfdetr import RFDETRSegPreview
            self.progress.emit("Loading RF-DETR…")
            rf = RFDETRSegPreview(pretrain_weights=RFDETR_PATH, device=DEVICE)
            rf.optimize_for_inference()
        except Exception as ex:
            self.load_errors["RF-DETR"] = str(ex)

        try:
            self.progress.emit("Loading DINOv3 checkpoint…")
            dino = DINOv3Model(DINOV3_PATH)
            if not dino.ready:
                self.load_errors["DINOv3"] = dino.error or "Invalid checkpoint"
                dino = None
        except Exception as ex:
            self.load_errors["DINOv3"] = str(ex)

        self.finished.emit(y12, y26, rf, dino)


class InferenceWorker(QThread):
    finished = pyqtSignal(list)

    def __init__(self, models, ip, conf, model_idx=0):
        super().__init__()
        self.models = models; self.ip = ip; self.conf = conf; self.model_idx = model_idx

    def run(self):
        bgr = cv2.imread(self.ip)
        if bgr is None:
            self.finished.emit([None]*5); return
        orig = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        orig_tup = (orig, "", (155, 160, 165), None)
        result = [orig_tup] + [None] * 4
        model = self.models[self.model_idx]
        if model is None:
            self.finished.emit(result)
            return

        if self.model_idx in (0, 1):
            prediction = model.predict(self.ip, conf=self.conf, imgsz=640, verbose=False)[0]
            result[self.model_idx + 1] = _draw_yolo(bgr, prediction, list(model.names.values()))
        elif self.model_idx == 2:
            prediction = model.predict(self.ip, threshold=self.conf, resolution=672)
            result[3] = _draw_rfdetr(bgr, prediction)
        else:
            prediction = model.predict(bgr)
            result[4] = _draw_dinov3(bgr, prediction, self.conf)

        self.finished.emit(result)


class GSThread(QThread):
    frame_ready  = pyqtSignal(np.ndarray)
    result_ready = pyqtSignal(np.ndarray, str, tuple, object)
    det_classes  = pyqtSignal(set)
    status       = pyqtSignal(str)

    def __init__(self, models, conf, midx=0, cam=GELSIGHT_CAM):
        super().__init__()
        self.models = models; self.conf = conf; self.midx = midx; self.cam = cam
        self._run = False; self._mx = QMutex(); self._nc = None; self._nm = None
        self._is_inferring = False

    def set_conf(self, v):
        with QMutexLocker(self._mx): self._nc = v
    def set_model(self, i):
        with QMutexLocker(self._mx): self._nm = i
    def stop(self): 
        self._run = False

    def _do_inference(self, frm, midx, conf):
        y12, y26, rf, dino = self.models
        try:
            tmp = "/tmp/_gs.png"; cv2.imwrite(tmp, frm); det = set()
            if midx == 0:
                r = y12.predict(tmp, conf=conf, imgsz=640, verbose=False)[0]
                if r.boxes is not None: det = {int(c) for c in r.boxes.cls.cpu().numpy()}
                drw, txt, col, mtr = _draw_yolo(frm, r, list(y12.names.values()))
            elif midx == 1:
                r = y26.predict(tmp, conf=conf, imgsz=640, verbose=False)[0]
                if r.boxes is not None: det = {int(c) for c in r.boxes.cls.cpu().numpy()}
                drw, txt, col, mtr = _draw_yolo(frm, r, list(y26.names.values()))
            elif midx == 2:
                r = rf.predict(tmp, threshold=conf, resolution=672)
                if r is not None and len(r) > 0: det = {int(c) for c in r.class_id}
                drw, txt, col, mtr = _draw_rfdetr(frm, r)
            else:
                prediction = dino.predict(frm)
                predicted_classes = prediction.argmax(axis=0)
                pixel_confidence = prediction.max(axis=0)
                det = set(np.unique(predicted_classes[pixel_confidence >= conf])) - {4}
                drw, txt, col, mtr = _draw_dinov3(frm, prediction, conf)
            self.result_ready.emit(drw, txt, col, mtr)
            self.det_classes.emit(det)
        except Exception as ex:
            self.status.emit(f"Inference error: {ex}")
        finally:
            self._is_inferring = False

    def run(self):
        self._run = True
        cap = cv2.VideoCapture(self.cam, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened(): 
            self.status.emit(f"ERROR: Cannot open cam {self.cam}"); return
            
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.status.emit(f"GelSight connected — {W}×{H}px")
        fi = 1/STREAM_FPS
        lft = lii = 0.0
        
        while self._run:
            ret, frm = cap.read()
            if not ret: 
                time.sleep(0.05); continue
            now = time.time()
            
            with QMutexLocker(self._mx):
                if self._nc is not None: self.conf = self._nc; self._nc = None
                if self._nm is not None: self.midx = self._nm; self._nm = None
                
            if now - lft >= fi:
                self.frame_ready.emit(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB).copy())
                lft = now
                
            if now - lii >= config.INFER_INTERVAL and not self._is_inferring:
                lii = now
                self._is_inferring = True
                threading.Thread(target=self._do_inference, args=(frm.copy(), self.midx, self.conf), daemon=True).start()
                
        cap.release()
        self.status.emit("Stream stopped.")


class ROS2Thread(QThread):
    frame_ready  = pyqtSignal(np.ndarray)
    result_ready = pyqtSignal(np.ndarray, str, tuple, object)
    cloud_ready  = pyqtSignal(np.ndarray, object)
    fusion_ready = pyqtSignal(object)
    pcl_ready = pyqtSignal(object)
    export_finished = pyqtSignal(str)
    det_classes  = pyqtSignal(set)
    status       = pyqtSignal(str)

    def __init__(self, models, conf, midx=0, topic='/camera/image_raw'):
        super().__init__()
        self.models = models; self.conf = conf; self.midx = midx; self.topic = topic
        self._run = False; self._mx = QMutex(); self._nc = None; self._nm = None
        self._cloud_condition = threading.Condition()
        self._clouds_by_stamp = {}
        self._images_by_stamp = {}
        self._inference_lock = threading.Lock()
        self._inference_condition = threading.Condition(self._inference_lock)
        self._pending_synchronized_inference = None
        self._is_inferring = False
        self._last_inference_started = 0.0
        self._export_lock = threading.Lock()
        self._export_request = None
        self._bridge = None
        if ROS2_AVAILABLE:
            self._bridge = CvBridge()

    def set_conf(self, v):
        with QMutexLocker(self._mx): self._nc = v
    def set_model(self, i):
        with QMutexLocker(self._mx): self._nm = i
    def stop(self): 
        self._run = False

    def request_export(self, output_root):
        with self._export_lock:
            if self._export_request is not None:
                return False
            self._export_request = output_root
            return True

    def _take_export_request(self):
        with self._export_lock:
            output_root = self._export_request
            self._export_request = None
            return output_root

    def _repair_model(self):
        if not hasattr(self, '_repair_bundle'):
            from repairability import load_model
            self._repair_bundle = load_model(config.REPAIR_MODEL_PATH)
        return self._repair_bundle

    def _do_inference(self, frm, midx, conf, point_cloud=None):
        y12, y26, rf, dino = self.models
        try:
            tmp = "/tmp/_ros2_gs.png"; cv2.imwrite(tmp, frm); det = set()
            if midx == 0:
                r = y12.predict(tmp, conf=conf, imgsz=640, verbose=False)[0]
                regions = extract_segmentation_regions(midx, r, frm.shape[:2], conf)
                if r.boxes is not None: det = {int(c) for c in r.boxes.cls.cpu().numpy()}
                drw, txt, col, mtr = _draw_yolo(
                    frm, r, list(y12.names.values()), point_cloud
                )
            elif midx == 1:
                r = y26.predict(tmp, conf=conf, imgsz=640, verbose=False)[0]
                regions = extract_segmentation_regions(midx, r, frm.shape[:2], conf)
                if r.boxes is not None: det = {int(c) for c in r.boxes.cls.cpu().numpy()}
                drw, txt, col, mtr = _draw_yolo(
                    frm, r, list(y26.names.values()), point_cloud
                )
            elif midx == 2:
                r = rf.predict(tmp, threshold=conf, resolution=672)
                regions = extract_segmentation_regions(midx, r, frm.shape[:2], conf)
                if r is not None and len(r) > 0: det = {int(c) for c in r.class_id}
                drw, txt, col, mtr = _draw_rfdetr(frm, r, point_cloud)
            else:
                prediction = dino.predict(frm)
                regions = extract_segmentation_regions(
                    midx, prediction, frm.shape[:2], conf
                )
                predicted_classes = prediction.argmax(axis=0)
                pixel_confidence = prediction.max(axis=0)
                det = set(np.unique(predicted_classes[pixel_confidence >= conf])) - {4}
                drw, txt, col, mtr = _draw_dinov3(
                    frm, prediction, conf, point_cloud
                )
            if point_cloud is not None:
                self.cloud_ready.emit(_render_pointcloud_depth(point_cloud), mtr)
                self.pcl_ready.emit(
                    render_pcl_normals_view(
                        frm, point_cloud, regions, CLASS_COLORS_RGB
                    )
                )
                self.fusion_ready.emit(
                    render_depth_segmentation_overlay(
                        frm, point_cloud, regions, CLASS_COLORS_RGB
                    )
                )
                output_root = self._take_export_request()
                if output_root is not None:
                    try:
                        capture_dir, exported_count, skipped_count = export_segmented_pointclouds(
                            frm,
                            point_cloud,
                            regions,
                            output_root,
                            config.SEGMENTATION_TITLES[midx],
                            CLASS_COLORS_RGB,
                            CLASS_NAMES_SHORT,
                            self._repair_model(),
                        )
                        message = (
                            f'Exported {exported_count} segmented PCD(s) and PFH/SHOT '
                            f'descriptors to {capture_dir}'
                        )
                        manifest = json.loads(
                            (Path(capture_dir) / 'manifest.json').read_text(encoding='utf-8')
                        )
                        verdicts = [
                            f"{i['class_name']}: "
                            f"{'REPAIRABLE' if i['repairable'] else 'NON-REPAIRABLE'} "
                            f"({i['repairable_probability'] * 100:.0f}%)"
                            for i in manifest['instances'] if 'repairable' in i
                        ]
                        if verdicts:
                            message += ' | ML: ' + '; '.join(verdicts)
                        if skipped_count:
                            message += f'; skipped {skipped_count} mask(s) with fewer than 12 valid points'
                        self.export_finished.emit(message)
                    finally:
                        self.export_finished.emit('')
            self.result_ready.emit(drw, txt, col, mtr)
            self.det_classes.emit(det)
        except Exception as ex:
            self.status.emit(f"Inference error: {ex}")
    @staticmethod
    def _pointcloud_xyz(msg):
        fields = {field.name: field for field in msg.fields}
        if any(name not in fields for name in ('x', 'y', 'z')):
            return None
        if any(fields[name].datatype != PointField.FLOAT32 for name in ('x', 'y', 'z')):
            return None

        point_dtype = np.dtype({
            'names': ['x', 'y', 'z'],
            'formats': [np.float32, np.float32, np.float32],
            'offsets': [fields[name].offset for name in ('x', 'y', 'z')],
            'itemsize': msg.point_step,
        })
        point_count = msg.height * msg.width
        if point_count == 0 or len(msg.data) < point_count * msg.point_step:
            return None

        points = np.frombuffer(msg.data, dtype=point_dtype, count=point_count)
        return np.column_stack((points['x'], points['y'], points['z'])).reshape(
            msg.height, msg.width, 3
        )

    def _pointcloud_callback(self, msg):
        try:
            points = self._pointcloud_xyz(msg)
            if points is None:
                return
            stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
            if not stamp_ns:
                return
            with self._cloud_condition:
                self._clouds_by_stamp[stamp_ns] = (
                    msg.header.frame_id, points, msg.height, msg.width
                )
                while len(self._clouds_by_stamp) > 8:
                    del self._clouds_by_stamp[next(iter(self._clouds_by_stamp))]
            pair = self._take_synchronized_pair(stamp_ns)
            if pair is not None:
                image, point_cloud = pair
                self._schedule_inference(
                    image, self.midx, self.conf, point_cloud, synchronized=True
                )
        except Exception as exc:
            self.status.emit(f'PointCloud2 error: {exc}')

    def _take_synchronized_pair(self, stamp_ns):
        if not stamp_ns:
            return None

        with self._cloud_condition:
            image_entry = self._images_by_stamp.get(stamp_ns)
            cloud_entry = self._clouds_by_stamp.get(stamp_ns)
            if image_entry is None or cloud_entry is None:
                return None
            del self._images_by_stamp[stamp_ns]
            del self._clouds_by_stamp[stamp_ns]
        image_frame_id, image = image_entry
        cloud_frame_id, points, cloud_height, cloud_width = cloud_entry
        if image_frame_id and cloud_frame_id and image_frame_id != cloud_frame_id:
            return None
        image_shape = image.shape[:2]
        if points.shape[:2] == image_shape:
            return image, points
        if (
            cloud_height == 1
            and points.size == image_shape[0] * image_shape[1] * 3
        ):
            return image, points.reshape(*image_shape, 3)
        return None

    def _cache_image(self, stamp_ns, frame_id, image):
        if not stamp_ns:
            return None
        with self._cloud_condition:
            self._images_by_stamp[stamp_ns] = (frame_id, image.copy())
            while len(self._images_by_stamp) > 64:
                del self._images_by_stamp[next(iter(self._images_by_stamp))]
        return self._take_synchronized_pair(stamp_ns)

    def _schedule_inference(
        self, image, midx, conf, point_cloud=None, synchronized=False
    ):
        job = (image, midx, conf, point_cloud)
        with self._inference_condition:
            if self._is_inferring:
                if synchronized:
                    self._pending_synchronized_inference = job
                    self._inference_condition.notify()
                return
            self._is_inferring = True
        threading.Thread(
            target=self._run_inference_jobs,
            args=(job,),
            daemon=True,
        ).start()

    def _run_inference_jobs(self, job):
        while job is not None:
            with self._inference_condition:
                interval = max(0.0, float(config.INFER_INTERVAL))
                next_allowed = self._last_inference_started + interval
                while time.monotonic() < next_allowed:
                    self._inference_condition.wait(
                        timeout=next_allowed - time.monotonic()
                    )
                    if self._pending_synchronized_inference is not None:
                        job = self._pending_synchronized_inference
                        self._pending_synchronized_inference = None
                self._last_inference_started = time.monotonic()

            self._do_inference(*job)
            with self._inference_condition:
                job = self._pending_synchronized_inference
                self._pending_synchronized_inference = None
                if job is None:
                    self._is_inferring = False
                else:
                    self._inference_condition.notify()

    def run(self):
        if not ROS2_AVAILABLE:
            self.status.emit("ERROR: ROS 2 (rclpy) not found.")
            return

        try:
            if not rclpy.ok():
                rclpy.init()
            node = Node('defect_detector_ui')
            self.status.emit(f"ROS 2 Node started — Subscribing to {self.topic}")
            
            def image_callback(msg):
                if not self._run: return
                
                try:

                    frm = self._bridge.imgmsg_to_cv2(msg, 'bgr8')
                    
                    with QMutexLocker(self._mx):
                        if self._nc is not None: self.conf = self._nc; self._nc = None
                        if self._nm is not None: self.midx = self._nm; self._nm = None

                    self.frame_ready.emit(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB).copy())

                    stamp_ns = (
                        msg.header.stamp.sec * 1_000_000_000
                        + msg.header.stamp.nanosec
                    )
                    pair = self._cache_image(
                        stamp_ns, msg.header.frame_id, frm
                    )
                    if pair is not None:
                        image, point_cloud = pair
                        self._schedule_inference(
                            image, self.midx, self.conf, point_cloud,
                            synchronized=True,
                        )
                    else:
                        self._schedule_inference(
                            frm.copy(), self.midx, self.conf
                        )
                except Exception as e:
                    self.status.emit(f"ROS Callback error: {e}")

            qos_profile = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1
            )
            node.create_subscription(ROSImage, self.topic, image_callback, qos_profile=qos_profile)
            node.create_subscription(
                PointCloud2, '/gs_mini_pcd', self._pointcloud_callback,
                qos_profile=qos_profile
            )
            self._run = True
            
            while self._run:
                rclpy.spin_once(node, timeout_sec=0.1)
                
            node.destroy_node()
            self.status.emit("ROS 2 Subscription stopped.")
        except Exception as ex:
            self.status.emit(f"ROS 2 Error: {ex}")

class ClassifyWorker(QThread):
    finished = pyqtSignal(dict)
    def __init__(self, rn, cnn, img_path):
        super().__init__(); self._rn = rn; self._cnn = cnn; self._p = img_path
    
    def _run_resnet(self, bgr):
        try:
            rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
            img  = cv2.resize(rgb, (160, 128))
            x    = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(DEVICE)
            with torch.no_grad(): probs = torch.softmax(self._rn(x)[0], 0).cpu().numpy()
            idx = int(probs.argmax())
            return RESNET_CLASS_NAMES.get(idx, str(idx)), float(probs[idx])
        except Exception: return "Error", 0.0
        
    def _run_cnn(self, bgr):
        try:
            import tensorflow as tf
            rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB); img = cv2.resize(rgb, (160, 128)).astype("float32") / 255.0
            probs = self._cnn.predict(img[np.newaxis], verbose=0)[0]
            idx   = int(probs.argmax())
            return CLASS_NAMES.get(idx, str(idx)), float(probs[idx])
        except Exception: return "Error", 0.0
        
    def run(self):
        bgr = cv2.imread(self._p)
        if bgr is None: 
            self.finished.emit({"resnet":("—",0.), "cnn":("—",0.)}); return
        res = {
            "resnet": self._run_resnet(bgr) if self._rn  else ("Not loaded", 0.),
            "cnn":    self._run_cnn(bgr)    if self._cnn else ("Not loaded", 0.),
        }
        self.finished.emit(res)


class ClassifyLoader(QThread):
    progress = pyqtSignal(str); finished = pyqtSignal(object, object); error = pyqtSignal(str)
    
    def run(self):
        rn = cnn = None
        
        try:
            import torchvision.models as tvm, torch.nn as nn
            self.progress.emit("Loading ResNet18…")
            rn = tvm.resnet18(weights=None)
            rn.fc = nn.Linear(rn.fc.in_features, 7)
            ck = torch.load(RESNET_PATH, map_location=DEVICE, weights_only=False)
            state_dict = ck.get("state_dict", ck)
            clean_dict = {k.replace("model.", "") if k.startswith("model.") else k: v for k, v in state_dict.items()}
            rn.load_state_dict(clean_dict, strict=True)
            rn.to(DEVICE)
            rn.eval()
        except Exception as ex: 
            self.progress.emit(f"ResNet load error: {ex}")
            
        try:
            import tensorflow as tf
            self.progress.emit("Loading CNN…")
            try:
                cnn = tf.keras.models.load_model(CNN_CLS_PATH, compile=False)
            except Exception:
                cnn = tf.keras.models.Sequential([
                    tf.keras.layers.Input(shape=(128,160,3)),
                    tf.keras.layers.Conv2D(32,(5,5),activation='relu',input_shape=(128,160,3)), tf.keras.layers.MaxPooling2D((2,2)),
                    tf.keras.layers.Conv2D(64,(3,3),activation='relu'), tf.keras.layers.MaxPooling2D((2,2)),
                    tf.keras.layers.Conv2D(128,(3,3),activation='relu'), tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(128,activation='relu'), tf.keras.layers.Dropout(0.5), tf.keras.layers.Dense(5,activation='softmax')
                ])
                cnn.build((None, 128, 160, 3))
                cnn.load_weights(CNN_CLS_PATH)
        except Exception as ex: 
            self.progress.emit(f"CNN load error: {ex}")
            
        self.finished.emit(rn, cnn)