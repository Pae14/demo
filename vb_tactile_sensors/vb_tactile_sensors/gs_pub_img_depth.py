import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError

import cv2
import numpy as np
import torch
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

		# image
		self.timer = self.create_timer(0.04, self.publish_image_pcd)
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


	def publish_image_pcd(self):

		# create img msg and publish
		img = self.dev.get_image()
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
		stamp = self.get_clock().now().to_msg()

		img_msg = self.cvbridge.cv2_to_imgmsg(img, encoding="passthrough")
		img_msg.header = Header(stamp=stamp, frame_id=self.frame_id)

		self.pub_img.publish(img_msg)

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



def main(args=None):

	rclpy.init(args=args)

	camera_publisher = CameraPublisher()

	rclpy.spin(camera_publisher)

	camera_publisher.destroy_node()

	rclpy.shutdown()


if __name__ == '__main__':
	main()
