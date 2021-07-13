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

def FR(A,B, A_feat, B_feat):    

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
        corres_idx0, corres_idx1 = fcgf_feature_matching(fcgf_feats0, fcgf_feats1)

        start_time = time()

        # 2. Filter by distances in feature space
        corres_idx0, corres_idx1, corres_idx0_orig, corres_idx1_orig = filter_pairs_by_distance_in_feature_space(fcgf_feats0, fcgf_feats1, corres_idx0, corres_idx1, xyz0)

    # 3. Perform RANSAC
    T = RANSAC_registration(pcd0,
                                    pcd1,
                                    corres_idx0,
                                    corres_idx1,
                                    2 * voxel_size,
                                    num_iterations=500*10**3)

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

    elapsed_time = time() - start_time

    return T, elapsed_time, pcd0, pcd1

def fcgf_feature_matching(F0, F1):
    nn_max_n = 250

    def knn_dist(f0, f1):
        # Fast implementation with torch.einsum()
        with torch.no_grad():      
            # L2 distance:
            #   dist2 = torch.sum(f0**2, dim=1).reshape([-1,1]) + torch.sum(f1**2, dim=1).reshape([1,-1]) -2*torch.einsum('ac,bc->ab', f0, f1)
            #   dist = dist2.clamp_min(1e-30).sqrt_()
            # Cosine distance:
            dist = 1-torch.einsum('ac,bc->ab', f0, f1)                  
            min_dist, ind = dist.min(dim=1, keepdim=True)      
        return ind
    
    N = len(F0)
    C = int(np.ceil(N / nn_max_n))
    stride = nn_max_n
    inds = []
    for i in range(C):
        with torch.no_grad():
            ind = knn_dist(F0[i * stride:(i + 1) * stride], F1)
            inds.append(ind)
    
    inds = torch.cat(inds)
    assert len(inds) == N

    corres_idx0 = torch.arange(len(inds)).long().squeeze()
    corres_idx1 = inds.long().squeeze().cpu()
    return corres_idx0, corres_idx1

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

