#!/usr/bin/python3
# -*- coding: utf-8 -*-
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

import sys, Ice, os
from PySide6 import QtWidgets, QtCore

Ice.loadSlice("-I ./src/ --all ./src/Calibration.ice")
import RoboCompCalibration

class img(list):
    def __init__(self, iterable=list()):
        super(img, self).__init__(iterable)

    def append(self, item):
        assert isinstance(item, byte)
        super(img, self).append(item)

    def extend(self, iterable):
        for item in iterable:
            assert isinstance(item, byte)
        super(img, self).extend(iterable)

    def insert(self, index, item):
        assert isinstance(item, byte)
        super(img, self).insert(index, item)

setattr(RoboCompCalibration, "img", img)


import calibrationI




class GenericWorker(QtCore.QObject):

    kill = QtCore.Signal()

    def __init__(self, mprx):
        super(GenericWorker, self).__init__()


        self.mutex = QtCore.QMutex(QtCore.QRecursiveMutex)
        self.Period = 30
        self.timer = QtCore.QTimer(self)

