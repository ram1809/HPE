#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
from specificworker import SpecificWorker

import poseI
import Ice

if __name__ == '__main__':
    ic = Ice.initialize(sys.argv)
    status = 0
    mprx = {}

    worker = SpecificWorker(mprx)

    adapter = ic.createObjectAdapter('Pose')
    adapter.add(poseI.PoseI(worker), ic.stringToIdentity('pose'))
    adapter.activate()

    while True:
        worker.lock.acquire()
        worker.compute()
        worker.lock.release()
        # time.sleep(0.033) # about 30 FPS
        time.sleep(0.001) # about 30 FPS

    if ic:
        ic.destroy()
