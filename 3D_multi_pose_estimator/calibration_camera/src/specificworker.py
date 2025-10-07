#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
#    Copyright (C) 2021 by Luis J. Manso
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

REALSENSE = True
ZED = False
import time, sys
import cv2
import platform
import sys
import os
import os.path
import copy
import numpy as np
import json
import PIL.Image
import time
import platform

from genericworker import *

sys.path.append('../')
from parameters import parameters

REALSENSE = ZED = False
OPENCV = True
assert int(REALSENSE) + int(ZED) + int(OPENCV), "Only one MUST be selected, either ZED or REALSENSE."


if REALSENSE is True:
    sys.path.append('/usr/local/lib/python3.6/pyrealsense2')
    # import pyrealsense2.pyrealsense2 as rs
if ZED is True:
    import pyzed.sl as sl


camera_identifier = os.getenv("CAMERA_IDENTIFIER")
if camera_identifier == "":
    hostname = platform.node()
else:
    hostname = camera_identifier


if OPENCV is True:
    cv2grabber = cv2.VideoCapture(f"/dev/video{camera_identifier}")
    cam_index = parameters.camera_names.index(camera_identifier)
    print(f"[{camera_identifier}]  [{cam_index}]")
    cv2grabber.set(cv2.CAP_PROP_FRAME_WIDTH, parameters.widths[cam_index])
    cv2grabber.set(cv2.CAP_PROP_FRAME_HEIGHT, parameters.heights[cam_index])

#from dt_apriltags import *



if REALSENSE:
    DEPTH_WIDTH = 848
    DEPTH_HEIGHT = 480
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, COLOUR_WIDTH, COLOUR_HEIGHT, rs.format.bgr8, 30)
    pipeline.start(config)
    profile = pipeline.get_active_profile()
    color_intrin = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    print(color_intrin)
    fx = color_intrin.fx
    fy = color_intrin.fy
    cx = color_intrin.ppx
    cy = color_intrin.ppy
elif ZED:
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 60
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print('Couldn\'t open ZED camera')
        exit(1)
    calibration_params = zed.get_camera_information().camera_configuration.calibration_parameters
    fx, fy, cx, cy = calibration_params.left_cam.fx, calibration_params.left_cam.fy, calibration_params.left_cam.cx, calibration_params.left_cam.cy
    print(f'fx:{fx}  fy:{fy}  cx:{cx}  cy:{cy}')
    left_image = sl.Mat()
    right_image = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()


COLOUR_WIDTH = parameters.widths[parameters.camera_names.index(hostname)]
COLOUR_HEIGHT = parameters.heights[parameters.camera_names.index(hostname)]
fx = parameters.fx[parameters.camera_names.index(camera_identifier)]
fy = parameters.fy[parameters.camera_names.index(camera_identifier)]
cx = parameters.cx[parameters.camera_names.index(camera_identifier)]
cy = parameters.cy[parameters.camera_names.index(camera_identifier)]



last_sent = time.time()

class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.host = platform.node()

 

    def __del__(self):
        print('SpecificWorker destructor')

    def setParams(self, params):
        return True

    def compute(self):
        while True:
            try:
                time.sleep(0.00001)
                if REALSENSE:
                    frames = pipeline.wait_for_frames()
                    color_frame = frames.get_color_frame()
                    color_image = np.asanyarray(color_frame.get_data())
                    grey = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
                    send_grey = grey.reshape(1, grey.shape[0]*grey.shape[1])[0].tolist()
                    self.calibration_proxy.sendData(self.host,
                                                    COLOUR_WIDTH, COLOUR_HEIGHT, send_grey,
                                                    DEPTH_WIDTH, DEPTH_HEIGHT, np.zeros((DEPTH_HEIGHT, DEPTH_WIDTH), dtype=np.float32),
                                                    [fx, fy, cx, cy]
                                                    )
                elif ZED:
                    if zed.grab(runtime_parameters) != sl.ERROR_CODE.SUCCESS:
                        print('Can\'t grab.')
                        continue
                    zed.retrieve_image(left_image, sl.VIEW.LEFT)
                    zed.retrieve_image(right_image, sl.VIEW.RIGHT)
                    left_grey = cv2.cvtColor(left_image.get_data(), cv2.COLOR_RGB2GRAY)
                    left_grey_send = left_grey.reshape(1, left_grey.shape[0]*left_grey.shape[1])[0].tolist()
                    left_grey_send_D = left_grey.reshape(1, left_grey.shape[0]*left_grey.shape[1])[0].astype(float).tolist()
                    right_grey = cv2.cvtColor(right_image.get_data(), cv2.COLOR_RGB2GRAY)

                    right_grey_send = right_grey.reshape(1, right_grey.shape[0]*right_grey.shape[1])[0].tolist()
                    right_grey_send_D = right_grey.reshape(1, right_grey.shape[0]*right_grey.shape[1])[0].astype(float).tolist()
                    self.calibration_proxy.sendData(self.host+'_l',
                                                    left_grey.shape[1], left_grey.shape[0], left_grey_send,
                                                    left_grey.shape[1], left_grey.shape[0], left_grey_send_D,
                                                    [fx, fy, cx, cy]
                                                    )
                    self.calibration_proxy.sendData(self.host+'_r',
                                                    right_grey.shape[1], right_grey.shape[0], right_grey_send,
                                                    right_grey.shape[1], right_grey.shape[0], right_grey_send_D,
                                                    [fx, fy, cx, cy]
                                                    )
                elif OPENCV:
                    ret, color_image = cv2grabber.read()
                    rgb_send   = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY).reshape(1, color_image.shape[0]*color_image.shape[1])[0].tolist()
                    depth_send = np.zeros((COLOUR_HEIGHT, COLOUR_WIDTH), dtype=np.float64).reshape(1, color_image.shape[0]*color_image.shape[1])[0].tolist()
                    
                    self.calibration_proxy.sendData(camera_identifier,
                                COLOUR_WIDTH, COLOUR_HEIGHT, rgb_send,
                                COLOUR_WIDTH, COLOUR_HEIGHT, depth_send,
                                [fx, fy, cx, cy]
                                )
    
            except Ice.ConnectionRefusedException as e:
                print(e)
                print('Cannot connect to host, waiting a few seconds...')
                time.sleep(1)
            except Ice.ConnectionLostException as e:
                print(e)
                print('Cannot connect to host, waiting a few seconds...')
                time.sleep(1)
            except Ice.UnknownException as e:
                print(e)
                time.sleep(1)
            except Ice.ConnectionTimeoutException as e:
                print(e)
                time.sleep(1)





