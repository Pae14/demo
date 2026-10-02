import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from cv_bridge import CvBridge, CvBridgeError

import cv2
import numpy as np
import open3d as o3d

import os

from vb_tactile_sensors import gsdevice, gs3drecon, pointcloud2open3d


class TactileSubscriber(Node):

	def __init__(self):
		
		super().__init__('tactile_subscriber')
		self.img_subscriber = self.create_subscription(
			Image, '/gs_mini_img', self.img_callback, 50)
		self.pcd_subscriber = self.create_subscription(
			PointCloud2, '/gs_mini_pcd', self.pcd_callback, 50)

		self.cvbridge = CvBridge()

		self.save_path_img = "/rosbags/tactile_data/images/10_05_triangle/"
		self.save_path_pcd = "/rosbags/tactile_data/pcds/10_05_triangle/"

		self.img_counter = 159000
		self.pcd_counter = 159000


	def img_callback(self, msg):

		tactile_img = self.cvbridge.imgmsg_to_cv2(msg, "passthrough")
		cv2.imwrite(os.path.join(self.save_path_img, f"{self.img_counter}.png"), tactile_img)
		self.img_counter += 1
		

	def pcd_callback(self, msg):

		points_np = np.array(list(pointcloud2open3d.read_points(msg)))
		
		o3d_pcd = o3d.geometry.PointCloud()
		o3d_pcd.points = o3d.utility.Vector3dVector(points_np)
		o3d.io.write_point_cloud(os.path.join(self.save_path_pcd, f"{self.pcd_counter}.pcd"),
						o3d_pcd)
		self.pcd_counter += 1
		


def main(args=None):
	
	rclpy.init(args=args)

	tactile_subscriber = TactileSubscriber()

	rclpy.spin(tactile_subscriber)

	tactile_subscriber.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
