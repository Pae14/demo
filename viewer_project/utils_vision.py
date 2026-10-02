import cv2
import math
import numpy as np
from config import *

# https://scikit-image.org/docs/stable/api/skimage.measure.html
# https://scikit-image.org/docs/stable/api/skimage.feature.html

def _compute_pointcloud_depth_metrics(mask_bool, point_cloud):
    empty_metrics = {
        'depth_mean_mm': None,
        'depth_p95_mm': None,
        'depth_max_mm': None,
        'depth_volume_mm3': None,
    }
    if point_cloud is None:
        return empty_metrics

    point_cloud = np.asarray(point_cloud, dtype=np.float32)
    if point_cloud.shape != (*mask_bool.shape, 3):
        return empty_metrics

    valid = np.isfinite(point_cloud).all(axis=2)
    defect = mask_bool.astype(bool) & valid
    if defect.sum() < 3:
        return empty_metrics

    ring = cv2.dilate(
        mask_bool.astype(np.uint8), np.ones((13, 13), dtype=np.uint8)
    ).astype(bool) & ~mask_bool.astype(bool) & valid
    if ring.sum() < 3:
        return empty_metrics

    ring_points = point_cloud[ring]
    defect_points = point_cloud[defect]
    plane = np.column_stack((ring_points[:, 0], ring_points[:, 1], np.ones(ring.sum())))
    if np.linalg.matrix_rank(plane) == 3:
        coefficients = np.linalg.lstsq(plane, ring_points[:, 2], rcond=None)[0]
        defect_plane = np.column_stack((
            defect_points[:, 0], defect_points[:, 1], np.ones(defect.sum())
        )) @ coefficients
    else:
        defect_plane = np.full(defect.sum(), np.median(ring_points[:, 2]))

    depths_mm = np.maximum(defect_plane - defect_points[:, 2], 0) * 1000.0
    column_spacing = np.linalg.norm(np.diff(point_cloud[:, :, :2], axis=1), axis=2)
    row_spacing = np.linalg.norm(np.diff(point_cloud[:, :, :2], axis=0), axis=2)
    column_spacing = column_spacing[np.isfinite(column_spacing) & (column_spacing > 0)]
    row_spacing = row_spacing[np.isfinite(row_spacing) & (row_spacing > 0)]
    if not len(column_spacing) or not len(row_spacing):
        return empty_metrics

    pixel_area_mm2 = float(np.median(column_spacing) * np.median(row_spacing) * 1e6)
    return {
        'depth_mean_mm': float(np.mean(depths_mm)),
        'depth_p95_mm': float(np.percentile(depths_mm, 95)),
        'depth_max_mm': float(np.max(depths_mm)),
        'depth_volume_mm3': float(np.sum(depths_mm) * pixel_area_mm2),
    }


def _render_pointcloud_depth(point_cloud):
    point_cloud = np.asarray(point_cloud, dtype=np.float32)
    if point_cloud.ndim != 3 or point_cloud.shape[2] != 3:
        raise ValueError('Point cloud must have shape (height, width, 3)')

    height_mm = point_cloud[:, :, 2] * 1000.0
    valid = np.isfinite(height_mm)
    if not valid.any():
        return np.zeros((*height_mm.shape, 3), dtype=np.uint8)

    low, high = np.percentile(height_mm[valid], (2, 98))
    if high <= low:
        high = low + 1e-6
    normalized = np.zeros(height_mm.shape, dtype=np.uint8)
    normalized[valid] = np.clip(
        (height_mm[valid] - low) * (255.0 / (high - low)), 0, 255
    ).astype(np.uint8)
    depth_bgr = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    depth_rgb = cv2.cvtColor(depth_bgr, cv2.COLOR_BGR2RGB)
    depth_rgb[~valid] = (24, 24, 24)
    cv2.putText(
        depth_rgb,
        f'Z height: {low:.2f}-{high:.2f} mm',
        (8, max(18, height_mm.shape[0] - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return depth_rgb


def _exceeds_repair_depth(metrics):
    threshold = MAX_REPAIR_DEPTH_MM
    depth_p95 = metrics.get('depth_p95_mm')
    return threshold is not None and depth_p95 is not None and depth_p95 > threshold


def _format_depth_metric(metrics, key):
    value = metrics.get(key)
    return f'{value:.3f}' if value is not None else '—'


def _compute_metrics(mask_bool, gray_img, point_cloud=None):
    h, w = mask_bool.shape
    mm2_per_pixel = (GS_FOV_W_MM * GS_FOV_H_MM) / (w * h)
    mm_per_pixel  = math.sqrt(mm2_per_pixel)

    area_px   = float(np.sum(mask_bool))
    area_mm2  = area_px * mm2_per_pixel

    eccentricity  = 0.0
    perimeter_mm  = 0.0
    compactation  = 0.0
    
    try:
        from skimage.measure import regionprops, label as sklabel
        lbl_img = sklabel(mask_bool.astype(np.uint8))
        props_list = regionprops(lbl_img)
        if props_list:
            props = max(props_list, key=lambda p: p.area)
            eccentricity = float(props.eccentricity)
            perimeter_mm = float(props.perimeter) * mm_per_pixel
            if props.perimeter > 0:
                compactation = (4 * math.pi * props.area) / (props.perimeter ** 2)
    except Exception:
        contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            perim_px = sum(cv2.arcLength(c, True) for c in contours)
            perimeter_mm = perim_px * mm_per_pixel
            if perim_px > 0:
                compactation = (4 * math.pi * area_px) / (perim_px ** 2)

    glcm_homogeneity = 0.0
    glcm_energy      = 0.0
    glcm_entropy     = 0.0
    
    try:
        from skimage.feature import graycomatrix, graycoprops
        ys, xs = np.where(mask_bool)
        if len(xs) >= 4:
            y1c, y2c = ys.min(), ys.max() + 1
            x1c, x2c = xs.min(), xs.max() + 1
            roi_gray = gray_img[y1c:y2c, x1c:x2c]
            roi_mask = mask_bool[y1c:y2c, x1c:x2c]
            roi_masked = roi_gray.copy()
            roi_masked[~roi_mask] = 0
            roi_small = roi_masked
            maxd = 128
            rh, rw = roi_masked.shape

            if max(rh, rw) > maxd:
                scale = maxd / max(rh, rw)
                roi_small = cv2.resize(roi_masked, (max(1, int(rw * scale)), max(1, int(rh * scale))), interpolation=cv2.INTER_AREA)
            
            glcm = graycomatrix(roi_small, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                                levels=256, symmetric=True, normed=True)
            glcm_homogeneity = float(np.mean(graycoprops(glcm, 'homogeneity')))
            glcm_energy      = float(np.mean(graycoprops(glcm, 'energy')))
            eps = 1e-10
            p_norm = glcm / (glcm.sum(axis=(0, 1), keepdims=True) + eps)
            glcm_entropy = float(-np.mean(np.sum(p_norm * np.log2(p_norm + eps), axis=(0, 1))))
    except Exception:
        pass

    metrics = {
        "area_mm2":         area_mm2,
        "perimeter_mm":     perimeter_mm,
        "eccentricity":     eccentricity,
        "compactation":     compactation,
        "glcm_homogeneity": glcm_homogeneity,
        "glcm_energy":      glcm_energy,
        "glcm_entropy":     glcm_entropy,
    }
    metrics.update(_compute_pointcloud_depth_metrics(mask_bool, point_cloud))
    return metrics

def _get_status(has_defect, is_repairable):
    if not has_defect:
        return "NO DEFECTS", (155, 160, 165)
    if is_repairable:
        return "REPAIRABLE", (78, 170, 114)
    return "NON-REPAIRABLE", (217, 79, 79)

def _draw_yolo(bgr, result, names, point_cloud=None):
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).copy()
    h, w = img.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    if result.masks is None:
        return img, "NO DEFECTS", (155, 160, 165), None

    is_repairable = True
    has_defect = False
    
    n_list, c_list = [], []
    area_list, perim_list, ecc_list, comp_list = [], [], [], []
    homo_list, ener_list, ent_list = [], [], []
    depth_mean_list, depth_p95_list = [], []
    depth_max_list, depth_volume_list = [], []
    first_color = (180, 180, 180)

    for cid in np.unique(result.boxes.cls.cpu().numpy()):
        
        if int(cid) == 4:
            continue
        
        idxs = np.where(result.boxes.cls.cpu().numpy() == cid)[0]
        idxs = idxs[np.argsort(-result.boxes.conf.cpu().numpy()[idxs])]
        acc = np.zeros((h, w), dtype=bool)
        
        for j in idxs[:2]:
            mn = cv2.resize(result.masks.data[j].cpu().numpy(), (w, h))
            mb = mn > 0.5
            metrics = _compute_metrics(mb, gray, point_cloud)
            
            area_mm2 = metrics["area_mm2"]
            perim_mm = metrics["perimeter_mm"]
            ecc = metrics["eccentricity"]
            energia = metrics["glcm_energy"]
            entropia = metrics["glcm_entropy"]

            has_defect = True
            defect_repairable = True
            
            # Lógica de reparabilidad: Si CUALQUIERA de las condiciones falla (OR), es NO REPARABLE
            if int(cid) == 3: # Material loss - Holes
                if area_mm2 > 20.5 and perim_mm > 18.6:
                    defect_repairable = False
            elif int(cid) == 1: # Crack
                if area_mm2 > 10.36 or perim_mm > 14.2 or ecc < 0.96 or entropia > 9.30:
                    defect_repairable = False
            elif int(cid) == 0: # Scratch-Abrasion
                if energia < 0.04 or entropia > 8.46:
                    defect_repairable = False
            elif int(cid) == 2: # Scratch-Friction
                if energia < 0.04 or entropia > 8.75:
                    defect_repairable = False
            if _exceeds_repair_depth(metrics):
                defect_repairable = False

            if not defect_repairable:
                is_repairable = False

            if acc.sum() > 0 and (acc & mb).sum() / max((acc | mb).sum(), 1) > 0.7:
                continue
            acc |= mb
            
            col = CLASS_COLORS_RGB.get(int(cid), (180, 180, 180))
            if not n_list: first_color = col
            
            ov = np.zeros_like(img)
            ov[acc] = col
            img = cv2.addWeighted(img, 1.0, ov, 0.40, 0)
            ys, xs = np.where(acc)
            if len(xs):
                x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
                cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)

                lbl_name = CLASS_NAMES_SHORT.get(int(cid), names[int(cid)]).upper()
                lbl_conf = int(float(result.boxes.conf[j]) * 100)
                
                n_list.append(lbl_name)
                c_list.append(f"{lbl_conf}%")
                area_list.append(f"{area_mm2:.1f}")
                perim_list.append(f"{metrics['perimeter_mm']:.1f}")
                ecc_list.append(f"{ecc:.2f}")
                comp_list.append(f"{metrics['compactation']:.2f}")
                homo_list.append(f"{metrics['glcm_homogeneity']:.2f}")
                ener_list.append(f"{energia:.2f}")
                ent_list.append(f"{entropia:.2f}")
                depth_mean_list.append(_format_depth_metric(metrics, 'depth_mean_mm'))
                depth_p95_list.append(_format_depth_metric(metrics, 'depth_p95_mm'))
                depth_max_list.append(_format_depth_metric(metrics, 'depth_max_mm'))
                depth_volume_list.append(_format_depth_metric(metrics, 'depth_volume_mm3'))

    last_metrics = None
    if has_defect and n_list:
        last_metrics = {
            "class_name": " | ".join(n_list),
            "conf": " | ".join(c_list),
            "area_mm2": " | ".join(area_list),
            "perimeter_mm": " | ".join(perim_list),
            "eccentricity": " | ".join(ecc_list),
            "compactation": " | ".join(comp_list),
            "glcm_homogeneity": " | ".join(homo_list),
            "glcm_energy": " | ".join(ener_list),
            "glcm_entropy": " | ".join(ent_list),
            "depth_mean_mm": " | ".join(depth_mean_list),
            "depth_p95_mm": " | ".join(depth_p95_list),
            "depth_max_mm": " | ".join(depth_max_list),
            "depth_volume_mm3": " | ".join(depth_volume_list),
            "color": first_color
        }

    st_text, st_col = _get_status(has_defect, is_repairable)
    return img, st_text, st_col, last_metrics


def _draw_dinov3(bgr, class_probabilities, confidence_threshold=0.25, point_cloud=None):
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).copy()
    h, w = img.shape[:2]
    class_probabilities = np.asarray(class_probabilities, dtype=np.float32)
    if class_probabilities.ndim != 3 or class_probabilities.shape[0] != 5:
        raise ValueError('DINOv3 must return probabilities for five viewer classes')

    probabilities = np.stack([
        cv2.resize(class_probabilities[cid], (w, h), interpolation=cv2.INTER_LINEAR)
        for cid in range(5)
    ])
    class_ids = probabilities.argmax(axis=0)
    pixel_confidence = probabilities.max(axis=0)
    class_ids[pixel_confidence < confidence_threshold] = 4

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    overlay = np.zeros_like(img)
    has_defect = False
    is_repairable = True
    n_list, c_list = [], []
    area_list, perim_list, ecc_list, comp_list = [], [], [], []
    homo_list, ener_list, ent_list = [], [], []
    depth_mean_list, depth_p95_list = [], []
    depth_max_list, depth_volume_list = [], []
    first_color = (180, 180, 180)

    for cid in range(4):
        class_mask = (class_ids == cid).astype(np.uint8)
        component_count, components, stats, _ = cv2.connectedComponentsWithStats(
            class_mask, connectivity=8
        )
        for component_id in range(1, component_count):
            mask = components == component_id
            metrics = _compute_metrics(mask, gray, point_cloud)
            area_mm2 = metrics['area_mm2']
            perim_mm = metrics['perimeter_mm']
            ecc = metrics['eccentricity']
            energy = metrics['glcm_energy']
            entropy = metrics['glcm_entropy']

            defect_repairable = True
            if cid == 3:
                if area_mm2 > 20.5 and perim_mm > 18.6:
                    defect_repairable = False
            elif cid == 1:
                if area_mm2 > 10.36 or perim_mm > 14.2 or ecc < 0.96 or entropy > 9.30:
                    defect_repairable = False
            elif cid == 0:
                if energy < 0.04 or entropy > 8.46:
                    defect_repairable = False
            elif cid == 2:
                if energy < 0.04 or entropy > 8.75:
                    defect_repairable = False
            if _exceeds_repair_depth(metrics):
                defect_repairable = False
            if not defect_repairable:
                is_repairable = False

            has_defect = True
            color = CLASS_COLORS_RGB.get(cid, (180, 180, 180))
            if not n_list:
                first_color = color
            overlay[mask] = color

            x, y, box_w, box_h = stats[component_id, :4]
            cv2.rectangle(img, (x, y), (x + box_w - 1, y + box_h - 1), color, 2)
            class_name = CLASS_NAMES_SHORT.get(cid, str(cid)).upper()
            confidence = float(probabilities[cid][mask].mean())
            n_list.append(class_name)
            c_list.append(f'{int(confidence * 100)}%')
            area_list.append(f'{area_mm2:.1f}')
            perim_list.append(f'{perim_mm:.1f}')
            ecc_list.append(f'{ecc:.2f}')
            comp_list.append(f"{metrics['compactation']:.2f}")
            homo_list.append(f"{metrics['glcm_homogeneity']:.2f}")
            ener_list.append(f'{energy:.2f}')
            ent_list.append(f'{entropy:.2f}')
            depth_mean_list.append(_format_depth_metric(metrics, 'depth_mean_mm'))
            depth_p95_list.append(_format_depth_metric(metrics, 'depth_p95_mm'))
            depth_max_list.append(_format_depth_metric(metrics, 'depth_max_mm'))
            depth_volume_list.append(_format_depth_metric(metrics, 'depth_volume_mm3'))

    if not has_defect:
        return img, 'NO DEFECTS', (155, 160, 165), None

    img = cv2.addWeighted(img, 1.0, overlay, 0.40, 0)
    last_metrics = {
        'class_name': ' | '.join(n_list),
        'conf': ' | '.join(c_list),
        'area_mm2': ' | '.join(area_list),
        'perimeter_mm': ' | '.join(perim_list),
        'eccentricity': ' | '.join(ecc_list),
        'compactation': ' | '.join(comp_list),
        'glcm_homogeneity': ' | '.join(homo_list),
        'glcm_energy': ' | '.join(ener_list),
        'glcm_entropy': ' | '.join(ent_list),
        'depth_mean_mm': ' | '.join(depth_mean_list),
        'depth_p95_mm': ' | '.join(depth_p95_list),
        'depth_max_mm': ' | '.join(depth_max_list),
        'depth_volume_mm3': ' | '.join(depth_volume_list),
        'color': first_color,
    }
    status_text, status_color = _get_status(has_defect, is_repairable)
    return img, status_text, status_color, last_metrics


# Backward compatible alias
_draw_dinov3_preview = _draw_dinov3


def _draw_rfdetr(bgr, preds, point_cloud=None):
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).copy()
    h, w = img.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    if preds is None or len(preds) == 0 or preds.mask is None:
        return img, "NO DEFECTS", (155, 160, 165), None

    is_repairable = True
    has_defect = False
    
    n_list, c_list = [], []
    area_list, perim_list, ecc_list, comp_list = [], [], [], []
    homo_list, ener_list, ent_list = [], [], []
    depth_mean_list, depth_p95_list = [], []
    depth_max_list, depth_volume_list = [], []
    first_color = (180, 180, 180)

    for cid in np.unique(preds.class_id):

        if int(cid) == 4:
            continue
        
        idxs = np.where(preds.class_id == cid)[0]
        idxs = idxs[np.argsort(-preds.confidence[idxs])]
        acc = np.zeros((h, w), dtype=bool)
        for i in idxs[:2]:
            mask = preds.mask[i].astype(bool)
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            metrics = _compute_metrics(mask, gray, point_cloud)
            
            area_mm2 = metrics["area_mm2"]
            perim_mm = metrics["perimeter_mm"]
            ecc = metrics["eccentricity"]
            energia = metrics["glcm_energy"]
            entropia = metrics["glcm_entropy"]

            has_defect = True
            defect_repairable = True
            
            # Lógica de reparabilidad: Si CUALQUIERA de las condiciones falla (OR), es NO REPARABLE
            if int(cid) == 3: # Material loss - Holes
                if area_mm2 > 20.5 or perim_mm > 18.6:
                    defect_repairable = False
            elif int(cid) == 1: # Crack
                if area_mm2 > 10.36 or perim_mm > 14.2 or ecc < 0.98 or entropia > 9.28:
                    defect_repairable = False
            elif int(cid) == 0: # Scratch-Abrasion
                if energia < 0.04 or entropia > 8.46:
                    defect_repairable = False
            elif int(cid) == 2: # Scratch-Friction
                if energia < 0.04 or entropia > 8.75:
                    defect_repairable = False
            if _exceeds_repair_depth(metrics):
                defect_repairable = False

            if not defect_repairable:
                is_repairable = False

            if acc.sum() > 0 and (acc & mask).sum() / max((acc | mask).sum(), 1) > 0.7:
                continue
            acc |= mask
            
            col = CLASS_COLORS_RGB.get(int(cid), (180, 180, 180))
            if not n_list: first_color = col
            
            ov = np.zeros_like(img)
            ov[acc] = col
            img = cv2.addWeighted(img, 1.0, ov, 0.40, 0)
            ys, xs = np.where(acc)
            if len(xs):
                x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
                cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)

                lbl_name = CLASS_NAMES_SHORT.get(int(cid), str(int(cid))).upper()
                lbl_conf = int(float(preds.confidence[i]) * 100)
                
                n_list.append(lbl_name)
                c_list.append(f"{lbl_conf}%")
                area_list.append(f"{area_mm2:.1f}")
                perim_list.append(f"{metrics['perimeter_mm']:.1f}")
                ecc_list.append(f"{ecc:.2f}")
                comp_list.append(f"{metrics['compactation']:.2f}")
                homo_list.append(f"{metrics['glcm_homogeneity']:.2f}")
                ener_list.append(f"{energia:.2f}")
                ent_list.append(f"{entropia:.2f}")
                depth_mean_list.append(_format_depth_metric(metrics, 'depth_mean_mm'))
                depth_p95_list.append(_format_depth_metric(metrics, 'depth_p95_mm'))
                depth_max_list.append(_format_depth_metric(metrics, 'depth_max_mm'))
                depth_volume_list.append(_format_depth_metric(metrics, 'depth_volume_mm3'))

    last_metrics = None
    if has_defect and n_list:
        last_metrics = {
            "class_name": " | ".join(n_list),
            "conf": " | ".join(c_list),
            "area_mm2": " | ".join(area_list),
            "perimeter_mm": " | ".join(perim_list),
            "eccentricity": " | ".join(ecc_list),
            "compactation": " | ".join(comp_list),
            "glcm_homogeneity": " | ".join(homo_list),
            "glcm_energy": " | ".join(ener_list),
            "glcm_entropy": " | ".join(ent_list),
            "depth_mean_mm": " | ".join(depth_mean_list),
            "depth_p95_mm": " | ".join(depth_p95_list),
            "depth_max_mm": " | ".join(depth_max_list),
            "depth_volume_mm3": " | ".join(depth_volume_list),
            "color": first_color
        }

    st_text, st_col = _get_status(has_defect, is_repairable)
    return img, st_text, st_col, last_metrics
