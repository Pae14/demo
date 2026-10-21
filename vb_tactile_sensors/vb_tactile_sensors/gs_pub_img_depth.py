import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError

import cv2
import numpy as np
import torch
import threading
import time
from vb_tactile_sensors import gsdevice, gs3drecon

import os
class CameraPublisher(Node):

	def __init__(self):
	
		super().__init__('image_and_pointcloud_publisher')

		self.pub_img = self.create_publisher(Image, '/gs_mini_img', 10)
		self.pcd_publisher = self.create_publisher(PointCloud2, '/gs_mini_pcd', 10)
		self.publisher_ = self.create_publisher(Image, "/gs_depth_image", 10)

		self.dev = gsdevice.Camera("Gelsight Mini")
		self.dev.connect()

		self.cvbridge = CvBridge()

		# pointcloud - load model
		package_share_directory = os.path.dirname(os.path.realpath(__file__))
		net_path = os.path.join(package_share_directory, "nnmini.pt")
		device = "cuda" if torch.cuda.is_available() else "cpu"
		self.nn = gs3drecon.Reconstruction3D(self.dev)
		self.nn.load_nn(net_path, device)

		# pointcloud - store data points
		mmpp = 0.0634  # mini gel 18x24mm at 240x320
		self.mpp = mmpp / 1000.
		self.frame_id = 'gelsightmini_camera_frame'
		self._depth_condition = threading.Condition()
		self._pending_depth_frame = None
		self._depth_worker_running = True
		self._depth_update_interval = 0.2
		self._last_depth_frame_queued = 0.0
		self._capture_lock = threading.Lock()
		self._capture_running = True
		self._latest_frame = None
		self._latest_frame_sequence = 0
		self._published_frame_sequence = 0
		self._capture_thread = threading.Thread(
			target=self._capture_frames,
			name='gelsight-camera-capture',
			daemon=True,
		)
		self._capture_thread.start()
		self._depth_worker = threading.Thread(
			target=self._process_depth_frames,
			name='gelsight-depth-worker',
			daemon=True,
		)
		self._depth_worker.start()
		self.timer = self.create_timer(1.0 / 30.0, self.publish_image)


	def publish_image(self):
		with self._capture_lock:
			if (
				self._latest_frame is None
				or self._latest_frame_sequence == self._published_frame_sequence
			):
				return
			img = self._latest_frame.copy()
			self._published_frame_sequence = self._latest_frame_sequence

		stamp = self.get_clock().now().to_msg()
		img_msg = self.cvbridge.cv2_to_imgmsg(img, encoding="bgr8")
		img_msg.header = Header(stamp=stamp, frame_id=self.frame_id)
		self.pub_img.publish(img_msg)

		now = time.monotonic()
		if now - self._last_depth_frame_queued < self._depth_update_interval:
			return
		self._last_depth_frame_queued = now

		with self._depth_condition:
			self._pending_depth_frame = (img.copy(), stamp)
			self._depth_condition.notify()

	def _capture_frames(self):
		while self._capture_running:
			img = self.dev.get_image()
			if img is None:
				time.sleep(0.01)
				continue
			with self._capture_lock:
				self._latest_frame = img
				self._latest_frame_sequence += 1


	def _process_depth_frames(self):
		while True:
			with self._depth_condition:
				self._depth_condition.wait_for(
					lambda: (
						self._pending_depth_frame is not None
						or not self._depth_worker_running
					)
				)
				if not self._depth_worker_running:
					return
				img, stamp = self._pending_depth_frame
				self._pending_depth_frame = None

			try:
				self._publish_depth_pcd(img, stamp)
			except Exception as exc:
				self.get_logger().error(f'Depth reconstruction failed: {exc}')


	def _publish_depth_pcd(self, img, stamp):
		depth_map = self.nn.get_depthmap(img, False)
		if depth_map.shape != img.shape[:2]:
			self.get_logger().error(
				f'Depth map shape {depth_map.shape} does not match image {img.shape[:2]}'
			)
			return

		height, width = depth_map.shape
		depth_map_m = np.clip(depth_map.astype(np.float32) * self.mpp, 0, None)
		rows, columns = np.indices((height, width), dtype=np.float32)
		point_cloud = np.stack((
			columns * self.mpp,
			rows * self.mpp,
			depth_map_m,
		), axis=-1)
		data = point_cloud.astype(np.float32, copy=False).tobytes()

		# create pcd msg and publish
		ros_dtype = PointField.FLOAT32
		dtype = np.float32
		itemsize = np.dtype(dtype).itemsize  # 32-bit float takes 4 bytes
		fields = [PointField(
			name=n, offset=i * itemsize, datatype=ros_dtype, count=1)
			for i, n in enumerate('xyz')]

		header = Header(stamp=stamp, frame_id=self.frame_id)

		self.pcd = PointCloud2(
			header=header,
			height=height,
			width=width,
			is_dense=False,
			is_bigendian=False,
			fields=fields,
			point_step=(itemsize * 3),  # every point consists of 3 floats32
			row_step=(itemsize * 3 * width),
			data=data
			)

		# publish pcd
		self.pcd_publisher.publish(self.pcd)
		
		
		# Publish a display-only depth image in the same pixel order as the cloud.
		max_depth_m = 0.0022
		img_clipped = 255.0 * (depth_map_m / max_depth_m)

		# Clipping para evitar overflow
		img_clipped = np.clip(img_clipped, 0, 255)

		# Convertir a uint8
		img_escalada = img_clipped.astype(np.uint8)

		depth_image = img_escalada.reshape((height, width))

		# Convertir a mensaje de ROS2
		ros_image = self.cvbridge.cv2_to_imgmsg(depth_image, encoding="mono8")
		ros_image.header = Header(stamp=stamp, frame_id=self.frame_id)

		# Publicar la imagen en ROS2
		self.publisher_.publish(ros_image)

	def destroy_node(self):
		self._capture_running = False
		self._capture_thread.join(timeout=2.0)
		if self._capture_thread.is_alive():
			if self.dev.cam is not None:
				self.dev.cam.release()
			self._capture_thread.join(timeout=1.0)
			if self._capture_thread.is_alive():
				self.get_logger().error('Camera capture thread did not stop cleanly')
		with self._depth_condition:
			self._depth_worker_running = False
			self._depth_condition.notify_all()
		self._depth_worker.join()
		if self.dev.cam is not None:
			self.dev.cam.release()
		super().destroy_node()


def main(args=None):

	rclpy.init(args=args)

	camera_publisher = CameraPublisher()

	try:
		rclpy.spin(camera_publisher)
	finally:
		camera_publisher.destroy_node()
		rclpy.shutdown()


if __name__ == '__main__':
	main()
