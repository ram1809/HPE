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

# \mainpage RoboComp::tracker_camera
#
# \section intro_sec Introduction
#
# Some information about the component...
#
# \section interface_sec Interface
#
# Descroption of the interface provided...
#
# \section install_sec Installation
#
# \subsection install1_ssec Software depencences
# Software dependences....
#
# \subsection install2_ssec Compile and install
# How to compile/install the component...
#
# \section guide_sec User guide
#
# \subsection config_ssec Configuration file
#
# <p>
# The configuration file...
# </p>
#
# \subsection execution_ssec Execution
#
# Just: "${PATH_TO_BINARY}/tracker_camera --Ice.Config=${PATH_TO_CONFIG_FILE}"
#
# \subsection running_ssec Once running
#
#
#

import sys
import traceback
import IceStorm
import time
import os
import copy
from termcolor import colored

from specificworker import *


    
if __name__ == '__main__':
    ic = Ice.initialize(sys.argv[1])

    mprx = dict()
    # Remote object connection for Calibration
    try:
        proxyString = ic.getProperties().getProperty('CalibrationProxy')
        try:
            basePrx = ic.stringToProxy(proxyString)
            calibration_proxy = RoboCompCalibration.CalibrationPrx.uncheckedCast(basePrx)
            mprx["CalibrationProxy"] = calibration_proxy
        except Ice.Exception:
            print('Cannot connect to the remote object (Calibration)', proxyString)
            #traceback.print_exc()
            status = 1
    except Ice.Exception as e:
        print(e)
        print('Cannot get CalibrationProxy property.')
        status = 1


    worker = SpecificWorker(mprx)

    while True:
        worker.compute()
        time.sleep(0.015)
    if ic:
        ic.destroy()

        
