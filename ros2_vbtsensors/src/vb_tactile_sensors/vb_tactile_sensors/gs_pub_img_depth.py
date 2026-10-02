import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError

import cv2
import numpy as np
import torch
from vb_tactile_sensors import gsdevice, gs3drecon

from pathlib import Path
import copy

class CameraPublisher(Node):

	def __init__(self):
	
		super().__init__('image_and_pointcloud_publisher')

		self.pub_img = self.create_publisher(Image, '/gs_mini_img', 10)
		self.pcd_publisher = self.create_publisher(PointCloud2, '/gs_mini_pcd', 10)

		self.dev = gsdevice.Camera("Gelsight Mini")
		self.dev.connect()

		# image
		self.timer = self.create_timer(0.04, self.publish_image_pcd)
		self.cvbridge = CvBridge()

		# pointcloud - load model
		net_path = str(Path(__file__).with_name("nnmini.pt"))
		self.nn = gs3drecon.Reconstruction3D(self.dev)
		device = "cuda" if torch.cuda.is_available() else "cpu"
		self.nn.load_nn(net_path, device)

		# pointcloud - store data points
		# camera resolution
		mmpp = 0.0634  # mini gel 18x24mm at 240x320
		self.mpp = mmpp / 1000.
		x = np.arange(self.dev.imgh) * self.mpp
		y = np.arange(self.dev.imgw) * self.mpp
		X, Y = np.meshgrid(x, y)
		self.points = np.zeros([self.dev.imgw * self.dev.imgh, 3])
		self.points[:, 0] = np.ndarray.flatten(X)
		self.points[:, 1] = np.ndarray.flatten(Y)
		# initialize points array with zero depth
		Z = np.zeros((self.dev.imgh, self.dev.imgw))

		self.points[:, 2] = np.ndarray.flatten(Z)


	def publish_image_pcd(self):

		# create img msg and publish
		img = self.dev.get_image()
		img_msg = self.cvbridge.cv2_to_imgmsg(img, encoding="passthrough")
		
		img_msg.header = Header()
		img_msg.header.stamp = self.get_clock().now().to_msg()
		img_msg.header.frame_id = "gelsightmini_camera_frame"

		self.pub_img.publish(img_msg)

		# create pcd msg and publish
		# get depthmap and transform it to meters per pixel
		depth_map = self.nn.get_depthmap(img, False)
		depth_map_mpp = copy.deepcopy(depth_map) * self.mpp
		self.points[:, 2] = np.ndarray.flatten(depth_map_mpp)

		# create pcd msg
		ros_dtype = PointField.FLOAT32
		dtype = np.float32
		itemsize = np.dtype(dtype).itemsize  # 32-bit float takes 4 bytes

		data = self.points.astype(dtype).tobytes()

		# the fields specify what the bytes represents. The first 4 bytes
		# represents the x-coordinate, the next 4 the y-coordinate, etc.
		fields = [PointField(
			name=n, offset=i * itemsize, datatype=ros_dtype, count=1)
			for i, n in enumerate('xyz')]

		header = Header(frame_id='gelsightmini_camera_frame')
		header.stamp = self.get_clock().now().to_msg()

		self.pcd = PointCloud2(
			header=header,
			height=1,
			width=self.points.shape[0],
			is_dense=False,
			is_bigendian=False,
			fields=fields,
			point_step=(itemsize * 3),  # every point consists of 3 floats32
			row_step=(itemsize * 3 * self.points.shape[0]),
			data=data
			)

		# publish pcd
		self.pcd_publisher.publish(self.pcd)


def main(args=None):

	rclpy.init(args=args)

	camera_publisher = CameraPublisher()

	rclpy.spin(camera_publisher)

	camera_publisher.destroy_node()

	rclpy.shutdown()


if __name__ == '__main__':
	main()
