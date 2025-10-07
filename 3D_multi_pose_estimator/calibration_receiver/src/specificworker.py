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

import os
import PySide6 as PySide2
dirname = os.path.dirname(PySide2.__file__)
plugin_path = os.path.join(dirname, 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QComboBox
from genericworker import *
import time
import numpy as np
import cv2
import json
import threading
import copy

from dt_apriltags import *

import matplotlib.pyplot as plt
from pytransform3d import rotations as pr
from pytransform3d import transformations as pt
from pytransform3d.transform_manager import TransformManager

sys.path.append('../')
from parameters import parameters

NUM_CAMERAS = len(parameters.cameras)
room_width = 7.
map_res = 900.
draw = True
TAG_SIZE=parameters.tag_size

class CameraData(object):
    def __init__(self, host, w, h, grey, dw, dh, depth, fxfycxcy):
        self.host = host
        self.cw = w
        self.ch = h
        self.dw = dw
        self.dh = dh
        self.fx, self.fy, self.cx, self.cy = fxfycxcy

        self.grey = np.array(copy.deepcopy(grey)).reshape(h, w).astype(np.uint8)
        #self.depth = np.array(copy.deepcopy(depth)).reshape(dh, dw).astype(np.float)

        # self.grey = cv2.resize(self.grey, (w//2, h//2))

        self.detector = Detector(searchpath=['apriltags'], families='tagStandard41h12',
                       nthreads=5, quad_decimate=1.0, quad_sigma=0.1, refine_edges=1,
                       decode_sharpening=0.25, debug=0)
        cam_prms = [self.fx, self.fy, self.cx, self.cy]
        print(f'cam_prms {cam_prms}')
        self.tags = self.detector.detect(self.grey, estimate_tag_pose=True, camera_params=cam_prms, tag_size=TAG_SIZE)
        print('tags', self.tags)
        if len(self.tags) > 0:
            t = self.tags[0].pose_t.ravel()#*0.567518248
            print(host, np.linalg.norm(t))
            self.transform = pt.transform_from(self.tags[0].pose_R, t)
        else:
            self.transform = None
        #self.tags = []
        #self.transform = None

class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = 1.
        self.selection = QComboBox()
        self.selection.resize(200, 90)
        self.selection.show()
        self.data_map = dict()
        self.data_map_raw = dict()
        self.lock = threading.RLock()
        self.last_apriltags_check = time.time()

        self.timer.timeout.connect(self.compute)
        self.timer.start(self.Period)

        self.room_map = np.full((int(map_res),int(map_res),3), 255, dtype=np.uint8)

    def __del__(self):
        print('SpecificWorker destructor')

    def draw_arrow(self, image, i_w, i_h, ratio, xa, ya, xC, yC, xb, yb):
        tlx = int(ratio*(xa))+int(i_w/2)
        tly = int(i_h)-int(ratio*(ya))
        brx = int(ratio*(xb))+int(i_w/2)
        bry = int(i_h)-int(ratio*(yb))
        mx = int(ratio*(xC))+int(i_w/2)
        my = int(i_h)-int(ratio*(yC))
    
        cv2.line(image, (tlx,tly), ( mx, my), (255,0,0), 3)
        cv2.line(image, ( mx, my), (brx,bry), (0,0,255), 3)

    @QtCore.Slot()
    def compute(self):
        try:
            self.lock.acquire()        
            # host = str(self.selection.currentText())
            # if len(host)==0:
            #     return
            for host in ["0", "2"]:
                c = CameraData(*self.data_map_raw[host])
                self.data_map[host] = c.transform

                self.tm = TransformManager()


                try:
                    # image = cv2.cvtColor(self.data_map[host].grey, cv2.COLOR_GRAY2BGR)
                    image = cv2.cvtColor(c.grey, cv2.COLOR_GRAY2BGR)
                    cv2.imwrite(f"{host}.png", c.grey)
                except Exception as e:
                    print(type(e), e)
                    sys.exit(1)

                for tag in c.tags: #self.data_map[host].tags:
                    for idx in range(len(tag.corners)):
                        cv2.line(image, tuple(tag.corners[idx-1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)), (0, 255, 0))
                        cv2.putText(image, str(tag.tag_id),
                            org=(tag.corners[0, 0].astype(int)+10,tag.corners[0, 1].astype(int)+10),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.8,
                            color=(0, 0, 255))

                for host, data in self.data_map.items():
                    # if data.transform is not None:
                    if data is not None:
                        self.tm.add_transform("root", host, data) #data.transform)

            # got_list = [datav for datak, datav in self.data_map.items() if datav.transform is not None]
            got_list = [datav for datak, datav in self.data_map.items() if datav is not None]
            print(f"{got_list=}")
            if len(got_list) == NUM_CAMERAS:
                print('GOT IT')
                print('GOT IT')
                print('GOT IT')
                import pickle
                pickle.dump(self.tm, open('tm.pickle', 'wb'))
                # for v in got_list:
                #     pickle.dump(v.grey, open('grey_'+v.host+'.pickle', 'wb'))
                #     pickle.dump(v.depth, open('depth_'+v.host+'.pickle', 'wb'))
    
            cv2.imshow("image", image)
            k = cv2.waitKey(1)
            if k%256 == 27:
                print("Escape hit, closing...")
                cv2.destroyAllWindows()
                sys.exit(0)

        except Exception as e:
            print('Something went wrong in the compute', e)
        finally:
            self.lock.release()
        return 

    #
    # IMPLEMENTATION of sendData method from Pose interface
    #
    def Calibration_sendData(self, host, w, h, colour, dw, dh, depth, fxfycxcy):
        try:
            print('send data starts 1', host)
            self.lock.acquire()
            print('send data starts 2', host)
            # c = CameraData(host, w, h, colour, dw, dh, depth, fxfycxcy)
            c = (host, w, h, colour, dw, dh, depth, fxfycxcy)
            print('cam created for', host)
            try:
                print('.')
                self.data_map_raw[host] = c
                print(',')
                if self.selection.findText(host) == -1:
                    self.selection.addItem(host)
                print('b')
            except Exception as e:
                print(type(e), e, 'Cannot add the element')
        except Exception as e:
            print(type(e), e, 'Cannot build CameraData')
        finally:
            print('send data stops', host)
            self.lock.release()

