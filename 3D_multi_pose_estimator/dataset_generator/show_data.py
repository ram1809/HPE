import sys
import numpy as np
import json
import cv2
import pickle
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import math

sys.path.append('../')
from parameters import parameters

if len(sys.argv)!=3:
    print('USAGE: python3 show_data.py file.json camera_name')
    exit()

cameras = ['trackera', 'trackerb', 'trackerc', 'trackerd']

if sys.argv[2] not in cameras:
    print(sys.argv[2], ' is not a valid camera name')
    exit()
else:
    camera_name = sys.argv[2]

plt.ion()
fig = plt.figure(figsize=(10,10))
ax = plt.axes(projection='3d')

input_data_from_a_file = json.loads(open(sys.argv[1], "rb").read())

tm = pickle.load(open(parameters.transformations_path, 'rb'))

def to_world(tm, p, source):
    return tm.get_transform(source, "root") @ p

color = dict()
color['trackera'] = (0, 0, 255)
color['trackerb'] = (0, 255, 0)
color['trackerc'] = (255, 0, 0)
color['trackerd'] = (255, 255, 0)


xdata = dict()
ydata = dict()
zdata = dict()

for sample in input_data_from_a_file:
    img = np.ones((480,848,3), dtype=np.uint8) * 255
    map = np.ones((800,800,3), dtype=np.uint8) * 255

    for cam in cameras:
        xdata[cam] = []
        ydata[cam] = []
        zdata[cam] = []

    for cam in sample.keys():

        # if cam != camera_name:
        #     continue

        trackera = json.loads(sample[cam][0])
        # print(img.shape)
        # print(trackera)
        for sk in trackera:
            if len(sk) < 5:
                continue
            print('-----------')
            three_points = dict()
            for joint in sk:
                if joint == '2' or joint == '5' or joint == '6':
                    three_points[joint] = sk[joint][2]
                    print(joint, sk[joint][2])
                img_coord = sk[joint][1]
                map_coord = sk[joint][2]
                y = img_coord[1]
                x = img_coord[0]
                # print(joint, img_coord)
                cv2.circle(img, (int(round(x)), int(round(y))), 1, color[cam], 2)
                cv2.putText(img, str(joint),
                            (int(round(x)) + 5, int(round(y))),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)

                if map_coord[3]==1.:
                    P = to_world(tm, np.asarray(map_coord), cam)                        
                    y = map_coord[2]*400./7 + 400.
                    x = map_coord[0]*400./7 + 400.
                    # print(joint, map_coord)
                    # cv2.circle(map, (int(round(x)), int(round(y))), 1, color[cam], 2)

                    y = P[1]*400./7 + 400.
                    x = P[0]*400./7 + 400.
                    if cam==camera_name:
                        xdata[cam].append(P[0])
                        ydata[cam].append(P[1])
                        zdata[cam].append(-P[2])

                    cv2.circle(map, (int(round(x)), int(round(y))), 1, color[cam], 2)
                    # cv2.putText(map, str(joint),
                    #             (int(round(x)) + 5, int(round(y))),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
            if len(three_points) == 3:
                dx = three_points['2'][0] - three_points['5'][0]
                dy = three_points['2'][1] - three_points['5'][1]
                dz = three_points['2'][2] - three_points['5'][2]
                d_l = math.sqrt(dx*dx + dy*dy + dz*dz)

                dx = three_points['2'][0] - three_points['6'][0]
                dy = three_points['2'][1] - three_points['6'][1]
                dz = three_points['2'][2] - three_points['6'][2]
                d_r = math.sqrt(dx*dx + dy*dy + dz*dz)

                print("left", d_l, "right", d_r)

                if math.fabs(d_l - d_r) > 0.1:
                    P = to_world(tm, np.asarray(three_points['2']), cam)                        

                    y = P[1]*400./7 + 400.
                    x = P[0]*400./7 + 400.
                    cv2.circle(map, (int(round(x)), int(round(y))), 1, (255, 255, 0), 10) #color[cam], 2)



    cv2.imshow("image", img)
    cv2.imshow("map", map)
    # Data for three-dimensional scattered points
    ax.cla()
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    ax.set_zlim([0, 3.])

    ax.scatter(np.array(xdata[camera_name]), np.array(ydata[camera_name]), np.array(zdata[camera_name]), color='Black')
    fig.canvas.draw()
    fig.canvas.flush_events()


    k = cv2.waitKey(10)
    if k%256 == 27:
        print("Escape hit, closing...")
        cv2.destroyAllWindows()
        sys.exit(0)


