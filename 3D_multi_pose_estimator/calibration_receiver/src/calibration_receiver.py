#!/usr/bin/env python3
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
#


import sys
import traceback
import IceStorm
import time
import os
import copy
import argparse
#from termcolor import colored
# Ctrl+c handling
import signal


import PySide6 as PySide2
dirname = os.path.dirname(PySide2.__file__)
plugin_path = os.path.join(dirname, 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path

from specificworker import *



#SIGNALS handler
def sigint_handler(*args):
    PySide2.QtWidgets.QApplication.quit()
    
if __name__ == '__main__':
    app = PySide2.QtWidgets.QApplication(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument('iceconfigfile', nargs='?', type=str, default='etc/config')

    args = parser.parse_args()

    ic = Ice.initialize(args.iceconfigfile)
    status = 0
    mprx = {}
    parameters = {}
    for i in ic.getProperties():
        parameters[str(i)] = str(ic.getProperties().getProperty(i))

    worker = SpecificWorker(mprx)

    adapter = ic.createObjectAdapter('Calibration')
    adapter.add(calibrationI.CalibrationI(worker), ic.stringToIdentity('calibration'))
    adapter.activate()

    signal.signal(signal.SIGINT, sigint_handler)
    app.exec_()

    if ic:
        ic.destroy()
