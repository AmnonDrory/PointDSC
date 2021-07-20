# Parts of this implementation were copied from the Deep Global Registration project, that carries the following copyright notice:
    # Copyright (c) Chris Choy (chrischoy@ai.stanford.edu) and Wei Dong (weidong@andrew.cmu.edu)

__doc__ = """
Implementation of the FCGF feature-space distance filtered RANSAC algorithm (FR).
"""

import numpy as np
import torch
import open3d as o3d
from copy import deepcopy
from time import time
from algorithms.matching import find_nn, nn_to_mutual, measure_inlier_ratio

def filter_pairs_by_distance_in_feature_space(fcgf_feats0, fcgf_feats1, corres_idx0, corres_idx1, xyz0):
    F0 = fcgf_feats0[corres_idx0,:]
    F1 = fcgf_feats1[corres_idx1,:]
    feat_dist = torch.sqrt(torch.sum((F0-F1)**2,axis=1))        

    NUM_QUADS = 10
    TOTAL_NUM = 10000 # in practice, about half is selected. 
    NUM_PER_QUAD = int(np.ceil(TOTAL_NUM/NUM_QUADS**2))
    def to_quads(X, NUM_QUADS):
        EPS = 10**-3
        m = torch.min(X)
        M = torch.max(X)
        X_ = (X - m) / (M-m+EPS)
        res = torch.floor(NUM_QUADS*X_)
        return res

    quadrant_i = to_quads(xyz0[:,0], NUM_QUADS)
    quadrant_j = to_quads(xyz0[:,1], NUM_QUADS)
    keep = np.zeros(len(feat_dist), dtype=bool)
    num_remaining_quads = NUM_QUADS**2
    num_remaining_samples = TOTAL_NUM          
    for qi in range(NUM_QUADS):
        for qj in range(NUM_QUADS):
            samples_per_quad = int(np.ceil(num_remaining_samples / num_remaining_quads))
            is_quad_mask = (quadrant_i == qi) & (quadrant_j == qj)  
            is_quad_inds = is_quad_mask.nonzero(as_tuple=True)[0]
            if len(is_quad_inds) > samples_per_quad:
                ord = torch.argsort(feat_dist[is_quad_mask])
                is_quad_inds = is_quad_inds[ord[:samples_per_quad]]
            keep[is_quad_inds] = True
            num_remaining_samples -= len(is_quad_inds)
            num_remaining_quads -= 1

    corres_idx0_orig = deepcopy(corres_idx0)
    corres_idx1_orig = deepcopy(corres_idx1)
    corres_idx0 = corres_idx0[keep]
    corres_idx1 = corres_idx1[keep]

    return corres_idx0, corres_idx1, corres_idx0_orig, corres_idx1_orig

def FR(A,B, A_feat, B_feat, args, T_gt):    

    voxel_size = 0.3
    
    def make_open3d_point_cloud(xyz):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        return pcd
    
    xyz0 = A
    xyz1 = B
    xyz0_np = xyz0.detach().cpu().numpy().astype(np.float64)
    xyz1_np = xyz1.detach().cpu().numpy().astype(np.float64)
    pcd0 = make_open3d_point_cloud(xyz0_np) 
    pcd1 = make_open3d_point_cloud(xyz1_np)

    device = 'cuda:%d' % torch.cuda.current_device()
    fcgf_feats0 = A_feat.to(device)
    fcgf_feats1 = B_feat.to(device)

    with torch.no_grad():

        # 1. Coarse correspondences
        corres_idx0, corres_idx1 = find_nn(fcgf_feats0, fcgf_feats1)
        num_pairs_init = len(corres_idx0)
        inlier_ratio_init = measure_inlier_ratio(corres_idx0, corres_idx1, pcd0, pcd1, T_gt, voxel_size)

        start_time = time()

        # 2. Filter by distances in feature space
        if args.mode == "DFR":
            corres_idx0, corres_idx1, corres_idx0_orig, corres_idx1_orig = filter_pairs_by_distance_in_feature_space(fcgf_feats0, fcgf_feats1, corres_idx0, corres_idx1, xyz0)
        elif args.mode == "MFR":
            corres_idx0_orig, corres_idx1_orig = corres_idx0, corres_idx1
            corres_idx0, corres_idx1 = nn_to_mutual(fcgf_feats0, fcgf_feats1, corres_idx0, corres_idx1)
        else:
            assert False, "when running with algo==RANSAC, must define mode to either DFR or MFR"

        filter_time = time() - start_time

        num_pairs_filtered = len(corres_idx0)
        inlier_ratio_filtered = measure_inlier_ratio(corres_idx0, corres_idx1, pcd0, pcd1, T_gt, voxel_size)

    if args.special == 'inlier_only':
        T = np.eye(4)
        elapsed_time = 0
        return T, elapsed_time, pcd0, pcd1, num_pairs_init, inlier_ratio_init, num_pairs_filtered, inlier_ratio_filtered


    start_time = time()

    ransac_iters = 500*10**3
    if args.iters is not None:
        ransac_iters = args.iters
    print(f"ransac_iters={ransac_iters}") # AD DEL
    # 3. Perform RANSAC
    T = RANSAC_registration(pcd0,
                                    pcd1,
                                    corres_idx0,
                                    corres_idx1,
                                    2 * voxel_size,
                                    num_iterations=ransac_iters)

    # 4. estimate motion using all inlier pairs:
    corres_idx0_ = corres_idx0_orig.detach().numpy()
    corres_idx1_ = corres_idx1_orig.detach().numpy()
    pcd0_trans = deepcopy(pcd0)
    pcd0_trans.transform(T)
    dist2 = np.sum((np.array(pcd0_trans.points)[corres_idx0_,:] - np.array(pcd1.points)[corres_idx1_,:])**2, axis=1)
    is_close = dist2 < (2*voxel_size)**2
    inlier_corres_idx0 = corres_idx0_[is_close]
    inlier_corres_idx1 = corres_idx1_[is_close]
    corres = np.stack((inlier_corres_idx0, inlier_corres_idx1), axis=1)
    corres_ = o3d.utility.Vector2iVector(corres)
    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    T = p2p.compute_transformation(pcd0, pcd1, corres_)

    algo_time = time() - start_time
    elapsed_time = filter_time + algo_time

    return T, elapsed_time, pcd0, pcd1, num_pairs_init, inlier_ratio_init, num_pairs_filtered, inlier_ratio_filtered


def RANSAC_registration(pcd0, pcd1, idx0, idx1,
                        distance_threshold, num_iterations):        

    corres = np.stack((idx0, idx1), axis=1)
    corres = o3d.utility.Vector2iVector(corres)

    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        pcd0, 
        pcd1,
        corres, 
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        max_correspondence_distance=distance_threshold, 
        ransac_n=4,
        checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength()],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration=num_iterations, confidence=0.9999)
    )

    return result.transformation

