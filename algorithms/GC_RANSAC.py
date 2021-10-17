from time import time
import pygcransac
import numpy as np

def GC_RANSAC(A,B, distance_threshold, num_iterations, spatial_coherence_weight=None, use_sprt=None):
    
    x1y1z1_= np.ascontiguousarray(A)
    x2y2z2_= np.ascontiguousarray(B)
    params = {
    'threshold': distance_threshold, # default: 1.0
    'conf': 0.999, # default: 0.99
    'spatial_coherence_weight': 0.0, # default: 0.975
    'max_iters': num_iterations, # default: 10000
    'use_sprt': True, # default: True
    'min_inlier_ratio_for_sprt': 0.1, # default: 0.1
    'sampler': 0, # default: 1, 0=RANSAC, 1=PROSAC
    'neighborhood': 0, # default: 0
    'neighborhood_size': 20, # default: 20
    }

    if spatial_coherence_weight is not None:
        params['spatial_coherence_weight'] = spatial_coherence_weight
    if use_sprt is not None:
        params['use_sprt'] = use_sprt        

    start_time = time()
    pose_T, mask = pygcransac.findRigidTransform(
        x1y1z1_,
        x2y2z2_,
        **params)

    if pose_T is None:
        pose_T = np.eye(4, dtype=np.float32)

    elapsed_time = time() - start_time
    return pose_T.T, elapsed_time
