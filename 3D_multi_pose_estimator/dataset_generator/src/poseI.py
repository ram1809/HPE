#
#    Copyright (C) 2021 by YOUR NAME HERE
#
#    This file is part of RoboComp
#
#    RoboComp is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    RoboComp is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
#

import sys, os, Ice

Ice.loadSlice("-I ./src/ --all ./src/Pose.ice")

from RoboCompPose import *

class PoseI(Pose):
    def __init__(self, worker):
        self.worker = worker


    def sendData(self, host, data, other, c):
        return self.worker.Pose_sendData(host, data, other)

    def sendImage(self, host, image, other, c):
        return self.worker.Pose_sendImage(host, image, other)
