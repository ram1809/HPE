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

fake = True
draw = True

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
sys.path.append('/usr/local/lib/python3.6/pyrealsense2')
import pyrealsense2 as rs
import time
import platform

hostname = platform.node()
from dt_apriltags import *

from genericworker import * 

COLOUR_WIDTH = 1280
COLOUR_HEIGHT = 720
DEPTH_WIDTH = 848
DEPTH_HEIGHT = 480

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, COLOUR_WIDTH, COLOUR_HEIGHT, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth,  DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.z16, 30)
pipeline.start(config)


profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()
color_intrin = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
print(color_intrin)
fx = color_intrin.fx
fy = color_intrin.fy
cx = color_intrin.ppx
cy = color_intrin.ppy

depth_to_color_extrin =  profile.get_stream(rs.stream.depth).as_video_stream_profile().get_extrinsics_to( profile.get_stream(rs.stream.color))
color_to_depth_extrin =  profile.get_stream(rs.stream.color).as_video_stream_profile().get_extrinsics_to( profile.get_stream(rs.stream.depth))

last_sent = time.time()

class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.host = platform.node()

        self.at_detector = Detector(searchpath=['apriltags'],
                       families='tagStandard41h12',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)

        while True:
            time.sleep(0.00001)
            if fake:
                self.message = 'fake'
            else:
                self.message = self.compute()

            try:
                self.calibration_proxy.sendData(self.host, self.message)
            except Ice.ConnectionRefusedException:
                # print('Cannot connect to host, waiting a few seconds...')
                time.sleep(1)
            except Ice.ConnectionLostException:
                # print('Cannot connect to host, waiting a few seconds...')
                time.sleep(1)


    def __del__(self):
        print('SpecificWorker destructor')

    def setParams(self, params):
        return True

    def compute(self):
        data_to_publish = []
        t = time.time()
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        grey = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
        depth_frame = frames.get_depth_frame()
        depth_data = np.asanyarray(depth_frame.get_data())
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        depth_image = cv2.cvtColor((depth_data/(10000./255.)).astype(np.uint8), cv2.COLOR_GRAY2BGR)



        def c2d(p):
            x,y = p
            depth_min = 0.01
            depth_max = 2
            depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            depth_intrin = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
            color_intrin = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            depth_to_color_extrin =  profile.get_stream(rs.stream.depth).as_video_stream_profile().get_extrinsics_to(profile.get_stream(rs.stream.color))
            color_to_depth_extrin =  profile.get_stream(rs.stream.color).as_video_stream_profile().get_extrinsics_to(profile.get_stream(rs.stream.depth))
            depth_point = rs.rs2_project_color_pixel_to_depth_pixel(depth_frame.get_data(), depth_scale, depth_min, depth_max, depth_intrin, color_intrin, depth_to_color_extrin, color_to_depth_extrin, [x,y])
            depth_point[0]= int(depth_point[0]+0.5)
            depth_point[1]= int(depth_point[1]+0.5)
            return depth_point

        def d2xyz(depth_point):
            depth_min = 0.01
            depth_max = 2
            depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            depth_intrin = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
            color_intrin = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            depth_to_color_extrin =  profile.get_stream(rs.stream.depth).as_video_stream_profile().get_extrinsics_to(profile.get_stream(rs.stream.color))
            color_to_depth_extrin =  profile.get_stream(rs.stream.color).as_video_stream_profile().get_extrinsics_to(profile.get_stream(rs.stream.depth))
            if np.any(depth_point == None):
                return [0,0,0]
            return rs.rs2_deproject_pixel_to_point(depth_intrin, depth_point, depth_scale*depth_data[depth_point[1], depth_point[0]])


        # for i in range(counts[0]):
        #     keypoints = get_keypoint(objects, i, peaks)
        #     left = keypoints[5]
        #     right = keypoints[6]
        #     centre = keypoints[17]
        #     if left[1] and left[2] and centre[1] and centre[2] and right[1] and right[2]:
        #         lx = round(left[2] * WIDTH * X_compress)
        #         ly = round(left[1] * HEIGHT * Y_compress)
        #         rx = round(right[2] * WIDTH * X_compress)
        #         ry = round(right[1] * HEIGHT * Y_compress)
        #         cx = round(centre[2] * WIDTH * X_compress)
        #         cy = round(centre[1] * HEIGHT * Y_compress)

        #         c = d2xyz(c2d([cx,cy]))
        #         l = d2xyz(c2d([lx,ly]))
        #         r = d2xyz(c2d([rx,ry]))
        #         data_to_publish.append([l, c, r])

        tags = self.at_detector.detect(grey, estimate_tag_calibration=True, camera_params=[fx, fy, cx, cy], tag_size=0.297-0.01)
        print(tags)
        if draw:
            for tag in tags:
                for idx in range(len(tag.corners)):
                    cv2.line(color_image, tuple(tag.corners[idx-1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)), (0, 255, 0))
                    cv2.putText(color_image, str(tag.tag_id),
                      org=(tag.corners[0, 0].astype(int)+10,tag.corners[0, 1].astype(int)+10),
                      fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                      fontScale=0.8,
                      color=(0, 0, 255))
            cv2.imshow('d', depth_image)
            cv2.imshow("test", color_image)
            k = cv2.waitKey(1)
            if k%256 == 27:
                print("Escape hit, closing...")
                cv2.destroyAllWindows()
                sys.exit(0)

        return json.dumps(data_to_publish)





