import rclpy 
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError

import cv2
from vb_tactile_sensors import digit

class CameraPublisher(Node):

	def __init__(self):

		super().__init__('camera_publisher')

		self.pub_img = self.create_publisher(Image, "/digit_img", 10)

		self.digit_sensor = digit.DIGIT(0)
		self.digit_sensor.connect()

		self.timer = self.create_timer(0.04, self.publish_image)
		self.cvbridge = CvBridge()

	def publish_image(self):

		digit_img = self.digit_sensor.get_frame()
		img_msg = self.cvbridge.cv2_to_imgmsg(digit_img, encoding="passthrough")

		img_msg.header = Header()
		img_msg.header.stamp = self.get_clock().now().to_msg()
		img_msg.header.frame_id = "gelsightmini_camera_frame"

		self.pub_img.publish(img_msg)


def main(args=None):

	rclpy.init(args=args)

	camera_publisher = CameraPublisher()

	rclpy.spin(camera_publisher)

	camera_publisher.destroy_node()

	rclpy.shutdown()


if __name__ == '__main__':
	main()
