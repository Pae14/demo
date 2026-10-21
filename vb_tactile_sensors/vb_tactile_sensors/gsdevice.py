import cv2
import numpy as np
import os
import re
import shutil
import subprocess


class _FFmpegCapture:
    def __init__(self, process, width, height):
        self.process = process
        self.width = width
        self.height = height
        self.frame_bytes = width * height * 3
        self._reported_exit = False

    def read(self):
        if self.process.stdout is None:
            return False, None
        raw_frame = self.process.stdout.read(self.frame_bytes)
        if len(raw_frame) != self.frame_bytes:
            if not self._reported_exit and self.process.poll() is not None:
                print(
                    f'FFmpeg camera capture exited with status '
                    f'{self.process.returncode}'
                )
                self._reported_exit = True
            return False, None
        frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(
            self.height, self.width, 3
        )
        return True, frame

    def release(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.process.stdout is not None:
            self.process.stdout.close()

def warp_perspective(img, corners, output_sz):
    TOPLEFT, TOPRIGHT, BOTTOMRIGHT, BOTTOMLEFT = corners
    WARP_H = output_sz[0]
    WARP_W = output_sz[1]
    points1 = np.float32([TOPLEFT, TOPRIGHT, BOTTOMRIGHT, BOTTOMLEFT])
    points2 = np.float32([[0, 0], [WARP_W, 0], [WARP_W, WARP_H], [0, WARP_H]])
    matrix = cv2.getPerspectiveTransform(points1, points2)
    result = cv2.warpPerspective(img, matrix, (WARP_W, WARP_H))
    return result

def resize_crop_mini(img, imgw, imgh):
    # remove 1/7th of border from each size
    border_size_x, border_size_y = int(img.shape[0] * (1 / 7)), int(np.floor(img.shape[1] * (1 / 7)))
    # keep the ratio the same as the original image size
    img = img[border_size_x+2:img.shape[0] - border_size_x, border_size_y:img.shape[1] - border_size_y]
    # final resize for 3d
    img = cv2.resize(img, (imgw, imgh))
    return img


class Camera:
    def __init__(self, dev_type):
        # variable to store data
        self.data = None
        self.name = dev_type
        #self.dev_id = get_camera_id(dev_type)
        self.imgw = 320 # this is for R1, R1.5 is 240
        self.imgh = 240 # this is for R1, R1.5 is 320
        self.cam = None
        self._frame_is_resized = False
        self.while_condition = 1

    def connect(self):

        dev = os.environ.get("CAMERA_DEVICE", "/dev/video0")
        ffmpeg = shutil.which('ffmpeg')
        if ffmpeg and (
            dev.startswith('/dev/video') or dev.isdigit()
        ):
            device_path = (
                f'/dev/video{dev}' if dev.isdigit() else dev
            )
            width, height = 320, 240
            crop_and_scale = (
                'crop=iw-2*floor(iw/7):ih-2*floor(ih/7)-2:'
                'floor(iw/7):floor(ih/7)+2,'
                f'scale={width}:{height}:flags=bilinear,format=bgr24'
            )
            self.cam = _FFmpegCapture(
                subprocess.Popen(
                    [
                        ffmpeg,
                        '-hide_banner',
                        '-loglevel', 'warning',
                        '-fflags', 'nobuffer',
                        '-thread_queue_size', '1',
                        '-f', 'v4l2',
                        '-input_format', 'mjpeg',
                        '-framerate', '25',
                        '-i', device_path,
                        '-vf', crop_and_scale,
                        '-vsync', '0',
                        '-f', 'rawvideo',
                        '-pix_fmt', 'bgr24',
                        'pipe:1',
                    ],
                    stdout=subprocess.PIPE,
                ),
                width,
                height,
            )
            self._frame_is_resized = True
            print(
                f'Camera capture: FFmpeg MJPEG from {device_path}, '
                f'25 FPS requested, output {width}x{height}'
            )
        else:
            video_index = dev.removeprefix("/dev/video")
            if dev.startswith("/dev/video") and video_index.isdigit():
                self.cam = cv2.VideoCapture(int(video_index), cv2.CAP_V4L2)
            else:
                self.cam = cv2.VideoCapture(dev)
            self.cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if self.cam is None or not self.cam.isOpened():
                print('Warning: unable to open video source: ', dev)
            elif not ffmpeg:
                print('Warning: ffmpeg not found; using OpenCV camera capture')
        self.imgw = 240
        self.imgh = 320

        return self.cam

    def get_raw_image(self):
        for i in range(10): ## flush out fist 100 frames to remove black frames
            ret, f0 = self.cam.read()
        ret, f0 = self.cam.read()
        if ret:
            if not self._frame_is_resized:
                f0 = resize_crop_mini(f0,self.imgh,self.imgw)
        else:
            print('ERROR! reading image from camera')

        self.data = f0
        return self.data


    def get_image(self):

        ret, f0 = self.cam.read()
        if ret:
            if not self._frame_is_resized:
                f0 = resize_crop_mini(f0, self.imgh, self.imgw)
        else:
            print('ERROR! reading image from camera!')

        self.data = f0
        return self.data

    def save_image(self, fname):
         cv2.imwrite(fname, self.data)

    def start_video(self):
        # the default while condition is set to 1, change it for R1.5
        while( self.while_condition ):
            frame = self.get_image()
            cv2.imshow('frame', frame)
            cv2.waitKey(0)

    def stop_video(self):
        cv2.destroyAllWindows()

