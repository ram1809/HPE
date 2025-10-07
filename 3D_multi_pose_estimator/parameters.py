import sys

from collections import namedtuple

FORMAT = 'COCO'  # 'COCO'  'BODY_25'

if FORMAT == 'BODY_25':
    NECK_ID = 1
    LEFTSHOULDER_ID = 5
    RIGHTSHOULDER_ID = 2
    JOINT_LIST = [x for x in range(25)]
elif FORMAT == 'COCO':
    NECK_ID = 17
    LEFTSHOULDER_ID = 5
    RIGHTSHOULDER_ID = 6
    JOINT_LIST = [x for x in range(18)]
else:
    raise Exception('Format not set correctly in parameters.py!')

fields = (
    'tag_size',
    'cameras',
    'camera_names',
    'widths',
    'heights',
    'fx',
    'fy',
    'cx',
    'cy',
    'r_s',
    'r_w',
    'c_s',
    'c_w',
    'kd0',
    'kd1',
    'kd2',
    'p1',
    'p2',
    'joint_list',
    'numbers_per_joint',
    'numbers_per_joint_for_loss',
    'transformations_path',
    'used_cameras',
    'used_cameras_skeleton_matching',
    'used_joints',
    'min_number_of_views',
    'no3d',
    'draw',
    'format',
    'neck_id',
    'leftshoulder_id',
    'rightshoulder_id',
    'dataset_generator_records_only_one_skeleton',
    'tracker_sends_only_one_skeleton',
    'graph_alternative',
    'camera_colours',
    'old_data_to_remove',
    'image_width',
    'image_height',
    'axes_3D'
)

TrackerParameters = namedtuple('TrackerParameters', fields, defaults=(None,) * len(fields))

CONFIGURATION = 'YOURS'  # values = {PANOPTIC, NEWLAB, OLDLAB}

if CONFIGURATION == 'YOURS':
    parameters = TrackerParameters(
        image_width=640,
        image_height=480,
        cameras=[0, 1],
        camera_names=['0', '2'],
        tag_size=0.45,  # 0.313,
        # fx  =              [424.86101478, 424.08621444, 425.35061892, 425.43102505],
        # fy  =              [425.31913919, 424.35178743, 425.92296923, 425.84745869],
        # cx =               [420.07507557, 420.53503877, 421.15377069, 420.93675768],
        # cy =               [237.26322826, 239.09107219, 245.26624895, 240.66948926],
        kd0=[0., 0., 0., 0., 0., 0., 0.],
        kd1=[0., 0., 0., 0., 0., 0., 0.],
        kd2=[0., 0., 0., 0., 0., 0., 0.],
        p1=[0., 0., 0., 0., 0., 0., 0.],
        p2=[0., 0., 0., 0., 0., 0., 0.],

        #          1              2_l               2_r       3         4             bot_l             bot_r
        widths =[640, 640],
        heights=[480, 480],
        fx=[340., 330.],
        fy=[340., 330.],
        cx=[324., 320],
        cy=[240., 242],

        #   updated        updated            updated         updated     updated       updated            updated
        r_s=[   0,             0],
        r_w=[ 480,           480],
        c_s=[   0,             0],
        c_w=[ 640,           640],

        joint_list=JOINT_LIST,
        numbers_per_joint=14,  # 1 (joint detected?) 2 (x)  3 (y)  4 (over the threshold?)  5 (certainty)
        numbers_per_joint_for_loss=4,  # 1 (joint detected?) 2 (x)  3 (y)  4 (over the threshold?)  5 (certainty)
        transformations_path='/home/rmunusamy/Music/3D_multi_pose_estimator/tm.pickle',
    
        
        used_cameras=['0', '2'],
        # ['orinbot_l', 'orinbot_r'],
        used_cameras_skeleton_matching=['0', '2'],
        # ['orinbot_l', 'orinbot_r'],
        used_joints=[x for x in range(18)],
        min_number_of_views=2,
        no3d=True,
        draw=True,
        format=FORMAT,
        neck_id=NECK_ID,
        leftshoulder_id=LEFTSHOULDER_ID,
        rightshoulder_id=RIGHTSHOULDER_ID,
        tracker_sends_only_one_skeleton=True,
        dataset_generator_records_only_one_skeleton=True,
        graph_alternative='3',
        camera_colours={'0': (255, 0, 0), '2': (0, 0, 255)},
        old_data_to_remove=0.08,
        axes_3D = {'X': (0, 1.), 'Y': (2, 1.), 'Z': (1, -1.)} #For drawing the skeletons: each tuple represents (coordinate index, axis direction)
    )
else:
    print('NO VALID CONFIGURATION')
    exit()

#
#  ASSERTS
#
assert len(parameters.cameras) == len(
    parameters.camera_names), "The number of cameras must be equal in 'cameras' and 'camera_names'"
assert NECK_ID in parameters.joint_list, f"Joint {NECK_ID} (neck) is mandatory"
