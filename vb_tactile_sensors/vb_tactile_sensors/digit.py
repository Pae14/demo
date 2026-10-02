import cv2

import logging
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

class DIGIT():

	def __init__(self, camera_device):

		self.camera_device = camera_device

		self.imgh = 240
		self.imgw = 320

		self.GEL_WIDTH = 0.01832
		self.GEL_HEIGHT = 0.02505
		self.MAX_DEPTH = 0.02076
		self.PROJ_MATRIX = [ [ 2.30940108e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00 ],
         [ 0.00000000e+00, 1.73205081e+00, 0.00000000e+00, 0.00000000e+00 ],
         [ 0.00000000e+00, 0.00000000e+00, -1.04081633e+00, -2.04081633e-03 ],
         [ 0.00000000e+00, 0.00000000e+00, -1.00000000e+00, 0.00000000e+00 ] ]

		self.SENSOR_PARAMS = OmegaConf.create({'gel_width':self.GEL_WIDTH, 'gel_height':self.GEL_HEIGHT, 'max_depth':self.MAX_DEPTH, 'fps':30,
                           'resolution':'QVGA', "sensor_proj_matrix":self.PROJ_MATRIX, 'z_near':0.001, 'z_far':0.05})


	def connect(self):

		logger.info("Connecting to DIGIT")
		self.__cam = cv2.VideoCapture(self.camera_device)

		if not self.__cam.isOpened():
			logger.error(
				f"Cannot open video capture device {self.camera_device}"
			)
			raise Exception(f"Error opening video stream: {self.camera_device}")

		self.__cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.imgw)
		self.__cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.imgh)


	def get_frame(self):

		ret, frame = self.__cam.read()

		if not ret:

			logger.error(
				f"Cannot retrieve frame data from {self.camera_device}"
			)
			raise Exception(
				f"Unable to grab frame from {self.camera_device}"
			)

		frame = cv2.transpose(frame, frame)
		return cv2.flip(frame, 0)

	def show_view(self, window_name):

		while True:
			
			frame = self.get_frame()
			cv2.imshow(f"{window_name}", frame)
			if cv2.waitKey(1) == 27:
				break

		cv2.destroyAllWindows()
