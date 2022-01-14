# +
import pickle
import numpy as np
import os, cv2

with open('../data/SVS dimension/dimensions_dict.pkl', 'rb') as f :
    dimension_lookup = pickle.load(f)

def stitch(svs_id, x_y_pair, patch_size = 512, step = 256, downsample = 512, prob = None):
    dimension = dimension_lookup[svs_id]
    patch_size = patch_size // downsample
    step = step // downsample
    mask = np.zeros((dimension[1] // downsample, dimension[0] // downsample)).astype('float32')
    count = np.zeros((dimension[1] // downsample, dimension[0] // downsample)).astype('float32')
    
    #mask = np.zeros((dimension[1], dimension[0])).astype('float32')
    for i, (x, y) in enumerate(x_y_pair):
        x, y = int(x) // downsample, int(y) // downsample
        patch = np.zeros((patch_size, patch_size))
        patch.fill(prob[i])
        if(prob == None):
            mask[y : y + (patch_size - step), x : x + (patch_size - step)] = 1
        else:
            last_time = (mask[y : y + (patch_size - step), x : x + (patch_size - step)]) * count[y : y + (patch_size - step), x : x + (patch_size - step)]
            mask[y : y + (patch_size - step), x : x + (patch_size - step)] = (last_time + prob[i] + 1) / (count[y : y + (patch_size - step), x : x + (patch_size - step)] + 1)
            mask[y+ (patch_size - step) : y + patch_size, x + (patch_size - step) : x + patch_size] = prob[i];
            count[y : y + (patch_size - step), x : x + (patch_size - step)] += 1
            count[y+ (patch_size - step) : y + patch_size, x + (patch_size - step) : x + patch_size] += 1
# 
    mask = cv2.resize(mask, (dimension[0] // 4, dimension[1] // 4))
    return mask



