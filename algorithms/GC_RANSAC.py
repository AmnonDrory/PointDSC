from time import time
try:
    import pygcransac
except Exception as E:
    print("Ignoring exception: " + str(E))
import numpy as np

def GC_RANSAC(A,B, distance_threshold, num_iterations, args, match_quality):
    
    x1y1z1_= np.ascontiguousarray(A)
    x2y2z2_= np.ascontiguousarray(B)
    params = { # AD TODO: most of these are overwritten by command line arguments. organize. 
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

    params['spatial_coherence_weight'] = args.spatial_coherence_weight
    params['use_sprt'] = args.use_sprt
    params['sampler'] = args.prosac
    params['conf'] = args.GC_conf
    if args.use_edge_len:
        params['use_sprt'] = True
        params['min_inlier_ratio_for_sprt'] = -1 # negative value signals that c++ code should use edge-len pre-emption

    if args.prosac:
        # sort from best quality to worst
        ord = np.argsort(-match_quality)
        x1y1z1_ = x1y1z1_[ord,:]
        x2y2z2_ = x2y2z2_[ord,:]

    start_time = time()
    pose_T, mask = pygcransac.findRigidTransform(
        x1y1z1_,
        x2y2z2_,
        **params)

    if pose_T is None:
        pose_T = np.eye(4, dtype=np.float32)

    elapsed_time = time() - start_time
    return pose_T.T, elapsed_time
