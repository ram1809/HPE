import sys

import pickle
import cv2

for fn in sys.argv[1:]:
    print(fn)
    with open(fn, 'rb') as fd:
        x = pickle.load(fd)
        print(type(x))
        print(x.shape)
        cv2.imwrite(fn+'.png', x)