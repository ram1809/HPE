import numpy as np

import itertools


import sys
sys.path.append('../')
from parameters import parameters 

import copy
import json
import random


def permutations_generator(available, data_augmentation, min_number_of_views = 1):
    if data_augmentation is False:
        yield available
        return

    available_np = np.array(available)
    for combination in itertools.product(range(2), repeat=len(available)):
        combination_np = np.array(combination)
        if (available_np-combination_np < 0).any():
            continue
        if np.sum(combination) < min_number_of_views:
            continue
        yield combination

def permutations_generator_random(available, data_augmentation, max_combinations_number = 1):
    yield available

    if data_augmentation is False:
        return

    available_np = np.array(available)
    combinations = []
    for combination in itertools.product(range(2), repeat=len(available)):
        combination_np = np.array(combination)
        if (available_np-combination_np < 0).any():
            continue
        if (available_np-combination_np == 0).all() or (combination_np == 0).all():
            continue
        combinations.append(combination)
    random.shuffle(combinations)
    combinations = combinations[:max_combinations_number-1]
    for combination in combinations:
        yield combination


def add_data_to_json(json_data, min_number_of_views = 1):
    new_json_data = []
    json_index = 0
    
    for data in json_data:  # FOR EACH SAMPLE IN A JSON FILE
        orig_json_index = json_index
        new_data = copy.deepcopy(data)
        flags = [0]*len(parameters.camera_names)

        for c in data:  # FOR EACH CAMERA IN A SAMPLE
            c_index = parameters.camera_names.index(c)
            cam_data = json.loads(data[c][0])
            if cam_data:
                flags[c_index] = 1
                new_data[c].append(orig_json_index)
            else:
                del new_data[c]
        if np.sum(flags) > 0:
            new_json_data.append(new_data)
            json_index += 1

            for combination in permutations_generator(flags, True, min_number_of_views):
                if tuple(flags) != combination:
                    # print("comb",combination)
                    new_data = copy.deepcopy(data)
                    for c_index, part in enumerate(combination):
                        c = parameters.camera_names[c_index]
                        if c in new_data.keys():
                            if part == 0:
                                del new_data[c]
                            else:
                                new_data[c].append(orig_json_index)
                    new_json_data.append(new_data)
                    # print('augm', new_data)
                    json_index += 1

    return new_json_data

# json_data = json.loads(open('data1.json', "rb").read())
# add_data_to_json(json_data)
