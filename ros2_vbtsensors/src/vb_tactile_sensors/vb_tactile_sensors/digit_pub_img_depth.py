import rclpy 
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError

import cv2
import numpy as np
from vb_tactile_sensors import digit

import os
import copy
import sys

sys.path.append("/gelsight_ros2/digit-depth/")

from digit_depth.third_party import geom_utils
from digit_depth.train.prepost_mlp import *


class CameraPublisher(Node):

	def __init__(self):

		super().__init__('image_and_pointcloud_publisher')

		self.pub_img = self.create_publisher(Image, "/digit_img", 10)
		self.pcd_publisher = self.create_publisher(PointCloud2, "/digit_pcd", 10)

		self.digit_sensor = digit.DIGIT(0)
		self.digit_sensor.connect()

		self.timer = self.create_timer(0.04, self.publish_image_pcd)
		self.cvbridge = CvBridge()

		# pointcloud - load model
		path = "/gelsight_ros2/ros2_vbtsensors/src/vb_tactile_sensors/vb_tactile_sensors"
		model_path = os.path.join(path, "model.ckpt")
		self.model = torch.load(model_path).to('cuda')
		self.model.eval()


	def publish_image_pcd(self):

		# create img msg and publish
		digit_img = self.digit_sensor.get_frame()
		img_msg = self.cvbridge.cv2_to_imgmsg(digit_img, encoding="passthrough")

		img_msg.header = Header()
		img_msg.header.stamp = self.get_clock().now().to_msg()
		img_msg.header.frame_id = "gelsightmini_camera_frame"

		self.pub_img.publish(img_msg)

		# pointcloud
		digit_img_prep = preproc_mlp(digit_img)
		digit_normals = self.model(digit_img_prep).detach().cpu().numpy()
		digit_normal_post, _ = post_proc_mlp(digit_normals)
		digit_gradx, digit_grady = geom_utils._normal_to_grad_depth(img_normal=digit_normal_post, gel_width=self.digit_sensor.GEL_WIDTH, gel_height=self.digit_sensor.GEL_HEIGHT, bg_mask=None)
		digit_img_depth = geom_utils._integrate_grad_depth(digit_gradx, digit_grady, boundary=None, bg_mask=None, max_depth=self.digit_sensor.MAX_DEPTH)
		
		digit_points3d = geom_utils.depth_to_pts3d(depth=digit_img_depth, P=torch.tensor(self.digit_sensor.PROJ_MATRIX), V=torch.eye(4), params=self.digit_sensor.SENSOR_PARAMS)
		digit_points3d = geom_utils.remove_background_pts(digit_points3d, bg_mask=None)
		digit_points3d = np.moveaxis(digit_points3d.numpy(), 0, 1)
		
		# create pcd msg
		ros_dtype = PointField.FLOAT32
		dtype = np.float32
		itemsize = np.dtype(dtype).itemsize  # 32-bit float takes 4 bytes

		data = digit_points3d.astype(dtype).tobytes()

		fields = [PointField(
			name=n, offset=i * itemsize, datatype=ros_dtype, count=1)
			for i, n in enumerate('xyz')]

		header = Header(frame_id='digit_camera_frame')
		header.stamp = self.get_clock().now().to_msg()

		self.pcd = PointCloud2(
			header=header,
			height=1,
			width=digit_points3d.shape[0],
			is_dense=False,
			is_bigendian=False,
			fields=fields,
			point_step=(itemsize * 3),  # every point consists of 3 float32
			row_step=(itemsize * 3 * digit_points3d.shape[0]),
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
