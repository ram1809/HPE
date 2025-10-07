#!/usr/bin/python3
#
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

from genericworker import GenericWorker
import time, sys
import numpy as np
import cv2
import json
import pickle
import threading
import signal

import matplotlib.pyplot as plt
# from pytransform3d import rotations as pr
# from pytransform3d import transformations as pt
#from pytransform3d.transform_manager import TransformManager

from datetime import datetime

sys.path.append('../')
from parameters import parameters

NUMBER_OF_JOINTS = len(parameters.joint_list)

draw = True

DO_CSV = False
WAIT_TIME = 500

time_string = datetime.now().strftime('%Y-%m-%d__%H:%M:%S')
if DO_CSV:
    output_file = open('tracker_data_'+time_string+'.csv', 'w')
raw_output_file = open('tracker_data_'+time_string+'.json', 'w')

absolute_start = time.time()

class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map):
        self.last_timestamp = time.time()
        if DO_CSV:
                self.all_data_received = ''
        self.all_raw_data_received = []
        super(SpecificWorker, self).__init__(proxy_map)
        self.current_data_map = dict()
        self.lock = threading.RLock()

        self.new_data = False

        self.picture = np.full((480,848,3), 255, dtype=np.uint8)

#        cv2.namedWindow("projection", cv2.WND_PROP_FULLSCREEN)
#        cv2.setWindowProperty("projection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("projection", self.picture)
        k = cv2.waitKey(1)
        if k%256 == 27:
            print("Escape hit, closing...")
            time.sleep(1)
            print("Escape hit, closing...")
            sys.exit(0)

    def compute(self):
        if self.new_data is False:
            if time.time() - self.last_timestamp > WAIT_TIME:
                if DO_CSV:
                    output_file.write(self.all_data_received)
                raw_output_file.write(json.dumps(self.all_raw_data_received))
                sys.exit(0)
            return
        else:
            self.new_data = False

        # Remove old data
        t = time.time()
        self.last_timestamp = t
        for key in list(self.current_data_map.keys()):
            if t - self.current_data_map[key][1] > parameters.old_data_to_remove:
                del self.current_data_map[key]

        if len(self.current_data_map) == 0:
            return

        # This 'if' is just to give us three seconds after we 'hit' enter
        if time.time() - absolute_start > 3:
            self.all_raw_data_received.append(dict(self.current_data_map))

        if draw:
            self.picture = np.full((480,848,3), 255, dtype=np.uint8)


        output_line = ''
        human = {}
        number_of_cameras_with_data = 0
        for k in parameters.camera_names:
            try:
                data, _timestamp, other = self.current_data_map[k]
                received_d = json.loads(data)
                # If there are multiple skeletons, we get the one with more data, as it's more likely to be real
                human = {}
                for received in received_d:
                    if draw:
                        for idx_i in parameters.joint_list:
                            idx = str(idx_i)
                            try:
                                joint_data = received[idx]
                                coord = tuple([int(round(xx)) for xx in joint_data[1:3]])
                                cv2.circle(self.picture, coord, 4, parameters.camera_colours[k], thickness=-1)
                            except KeyError:
                                pass
                    if len(received) > len(human):
                        human = received

                if len(human)>0:
                    number_of_cameras_with_data += 1

                # CSV part
                for idx_i in parameters.joint_list:
                    idx = str(idx_i)
                    try:
                        joint_data = human[idx]
                    except KeyError:
                        joint_data = [0] * parameters.numbers_per_joint
                    output_line += ','.join([str(x) for x in joint_data])+', '

            except KeyError:
                output_line += '0,'*(NUMBER_OF_JOINTS*parameters.numbers_per_joint) + ' '
                continue


            # # Drawing part
            # if draw:
            #     lstr = str(parameters.leftshoulder_id)
            #     nstr = str(parameters.neck_id)
            #     rstr = str(parameters.rightshoulder_id)
            #     if lstr in human.keys() and nstr in human.keys() and rstr in human.keys():
            #         l, c, r = human[lstr][1:3], human[nstr][1:3], human[rstr][1:3]

            #         coord = tuple([int(round(xx)) for xx in human[nstr][1:3]])
            #         cv2.circle(self.picture, coord, 3, (255, 0,0))

        cv2.putText(self.picture, str(number_of_cameras_with_data), (40,40), cv2.FONT_HERSHEY_SIMPLEX, 1., (0,0,0), 1, cv2.LINE_AA) 


        # output_line += other

        # Data saving part
        if DO_CSV:
            output_line = output_line.strip()[:-1]
            self.all_data_received += output_line+'\n'
        cv2.imshow("projection", self.picture)
        k = cv2.waitKey(1)
        if k%256 == 27:
            if DO_CSV:
                output_file.write(self.all_data_received)
            raw_output_file.write(json.dumps(self.all_raw_data_received))
            print("Escape hit, closing...")
            cv2.destroyAllWindows()
            sys.exit(0)

        return True



    # =============== Methods for Component Implements ==================
    # ===================================================================

    #
    # IMPLEMENTATION of sendData method from Pose interface
    #
    def Pose_sendData(self, host, data, other):
        self.lock.acquire()
        self.current_data_map[host] = [data, time.time(), other]
        self.new_data = True
        self.lock.release()

    #
    # IMPLEMENTATION of sendImage method from Pose interface
    #
    def Pose_sendImage(self, host, image):
        pass

    # ===================================================================
    # ===================================================================


