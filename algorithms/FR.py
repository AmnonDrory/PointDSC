# Parts of this implementation were copied from the Deep Global Registration project, that carries the following copyright notice:
    # Copyright (c) Chris Choy (chrischoy@ai.stanford.edu) and Wei Dong (weidong@andrew.cmu.edu)

__doc__ = """
Implementation of the fast RANSAC algorithms DFR (FCGF feature-space distance filtered) and MFR (mutual-nearest-neighbor filtered).
"""

import numpy as np
import torch
import open3d as o3d
from copy import deepcopy
from time import time
from algorithms.matching import find_nn, nn_to_mutual, measure_inlier_ratio, find_2nd
from algorithms.GC_RANSAC import GC_RANSAC

def calc_distances_in_feature_space(fcgf_feats0, fcgf_feats1, corres_idx0, corres_idx1, args):
    eps = 10**-6
    if args.use_dist_ratio:
        idx0_2nd, idx1_2nd = find_2nd(fcgf_feats0, fcgf_feats1)
        assert (idx0_2nd == corres_idx0).all()
        
        A = fcgf_feats0[corres_idx0,:]
        B_1 = fcgf_feats1[corres_idx1,:]
        B_2 = fcgf_feats1[idx1_2nd,:]
        dist_1 = torch.sqrt(torch.sum((A-B_1)**2,axis=1))        
        dist_2 = torch.sqrt(torch.sum((A-B_2)**2,axis=1))        
        feat_dist = dist_1 / (dist_2+eps)

    else:
        F0 = fcgf_feats0[corres_idx0,:]
        F1 = fcgf_feats1[corres_idx1,:]
        feat_dist = torch.sqrt(torch.sum((F0-F1)**2,axis=1))        
    
    return feat_dist

def mark_best_buddies(fcgf_feats0, fcgf_feats1, corres_idx0, corres_idx1):
    bb_idx0, bb_idx1 = nn_to_mutual(fcgf_feats0, fcgf_feats1, corres_idx0, corres_idx1)
    
    corres_idx0_np = corres_idx0.detach().cpu().numpy()
    corres_idx1_np = corres_idx1.detach().cpu().numpy()
    bb_idx0_np = bb_idx0.detach().cpu().numpy()
    bb_idx1_np = bb_idx1.detach().cpu().numpy()

    P = 1 + np.max(corres_idx0_np)
    bb_idx_flat = P*bb_idx1_np + bb_idx0_np
    corres_idx_flat = P*corres_idx1_np + corres_idx0_np
    is_bb = np.in1d(corres_idx_flat,bb_idx_flat)
    num_bb = is_bb.sum()
    return is_bb, num_bb


def filter_pairs_BFR(fcgf_feats0, fcgf_feats1, corres_idx0, corres_idx1, xyz0, args):
    # cobination of DFR and MFR.
    # first step - take the mutual-nearest-neighbors as a core
    # second step - use a spatial grid to add more pairs, so that every cell has some representatives.
                    # Seelct additional representatives by distances in feature space 
    
    
    is_bb, num_bb = mark_best_buddies(fcgf_feats0, fcgf_feats1, corres_idx0, corres_idx1)
    
    GRID_WID = args.BFR_grid_wid
    TOTAL_NUM = args.BFR_factor*num_bb
    if args.BFR_strict and (TOTAL_NUM < num_bb):
        TOTAL_NUM = num_bb

    if args.BFR_ignore_bb:
        assert not args.BFR_strict, "BFR_ignore_bb and BFR_strict are incompatible"
        is_bb[:] = False
        num_bb = 0


    feat_dist = calc_distances_in_feature_space(fcgf_feats0, fcgf_feats1, corres_idx0, corres_idx1, args)
    def normalize(tens):
        m = torch.min(tens)
        M = torch.max(tens)
        return (tens - m)/(M-m)

    norm_feat_dist = normalize(feat_dist)

    if not args.BFR_strict:
        # when not in strict mode, we don't necessarily keep all best buddies. 
        # instead, when selecting the pairs for each quad, we first select from the 
        # best-buddies, ordered by feature distance. When we've taken all best buddies, 
        # we select from the others, separately ordered by feature distance. 
        # all of this is achieved by adding for all best-buddies an offset of -1
        # to their normalized feature-distances. After that, normalized feature distances
        # are in the range [-1,0] for best-buddies, and [0,1] for others. This ensures 
        # that in each cell, best-buddies are selected first, before other pairs are considered.
        norm_feat_dist[is_bb] -= 1

    def to_quads(X, GRID_WID):
        EPS = 10**-3
        m = torch.min(X)
        M = torch.max(X)
        X_ = (X - m) / (M-m+EPS)
        res = torch.floor(GRID_WID*X_)
        return res

    # 1. Count for each quad the number of pairs and best-buddies in it
    quadrant_i = to_quads(xyz0[corres_idx0,0], GRID_WID).detach().cpu().numpy()
    quadrant_j = to_quads(xyz0[corres_idx0,1], GRID_WID).detach().cpu().numpy()    
    min_per_quad = np.zeros([GRID_WID,GRID_WID]) + np.nan
    max_per_quad = np.zeros([GRID_WID,GRID_WID]) + np.nan
    
    for qi in range(GRID_WID):
        for qj in range(GRID_WID):            
            is_quad_mask = (quadrant_i == qi) & (quadrant_j == qj)  
            min_per_quad[qi,qj] = is_bb[is_quad_mask].sum()
            max_per_quad[qi,qj] = is_quad_mask.sum()

    if not args.BFR_strict:
        min_per_quad *= 0

    # 2. Calculate number-per-quad by approximate water-filling: 
    def apply_height(height):
        is_dwarf = max_per_quad < height
        is_giant = height < min_per_quad
        is_mid = ~is_dwarf & ~is_giant
        per_quad = is_dwarf*max_per_quad + is_giant*min_per_quad + is_mid*height
        return per_quad
    
    max_height = TOTAL_NUM
    min_height = 0
    steps  = 0
    curr_height = (max_height + min_height) / 2
    while (np.abs(max_height-min_height)>2):
        
        per_quad = apply_height(curr_height)
        cur_total = per_quad.sum()
        
        if cur_total == TOTAL_NUM:
            break
        elif cur_total < TOTAL_NUM:
            min_height = curr_height
        elif cur_total > TOTAL_NUM:
            max_height = curr_height
        
        curr_height = (max_height + min_height) / 2
        steps += 1

    per_quad = apply_height(np.round(curr_height))


    # 3. Select pairs for each quad. First take best-buddies, then
    #    take the pairs with the closest feature-space distance.
    keep = np.zeros(len(norm_feat_dist), dtype=bool)    
    if args.BFR_strict:
        keep[is_bb] = True

    for qi in range(GRID_WID):
        for qj in range(GRID_WID):            
            extra_per_quad = int(per_quad[qi,qj] - min_per_quad[qi,qj])
            if extra_per_quad > 0:
                is_quad_mask = (quadrant_i == qi) & (quadrant_j == qj)  
                if args.BFR_strict:
                    is_cand = is_quad_mask & ~is_bb        
                else:
                    is_cand = is_quad_mask
                if per_quad[qi,qj] == max_per_quad[qi,qj]:
                    keep[is_cand] = True
                else:
                    ord = torch.argsort(norm_feat_dist[is_cand]).detach().cpu().numpy()
                    is_cand_inds = is_cand.nonzero()[0]
                    keep_inds = is_cand_inds[ord[:extra_per_quad]]
                    keep[keep_inds] = True

    corres_idx0_orig = deepcopy(corres_idx0)
    corres_idx1_orig = deepcopy(corres_idx1)
    corres_idx0 = corres_idx0[keep]
    corres_idx1 = corres_idx1[keep]

    return corres_idx0, corres_idx1, corres_idx0_orig, corres_idx1_orig    

def filter_pairs_by_distance_in_feature_space(fcgf_feats0, fcgf_feats1, corres_idx0, corres_idx1, xyz0, args):
    
    feat_dist = calc_distances_in_feature_space(fcgf_feats0, fcgf_feats1, corres_idx0, corres_idx1, args)

    GRID_WID = 10
    TOTAL_NUM = 10000 # in practice, about half is selected. 
    NUM_PER_QUAD = int(np.ceil(TOTAL_NUM/GRID_WID**2))
    def to_quads(X, GRID_WID):
        EPS = 10**-3
        m = torch.min(X)
        M = torch.max(X)
        X_ = (X - m) / (M-m+EPS)
        res = torch.floor(GRID_WID*X_)
        return res

    quadrant_i = to_quads(xyz0[:,0], GRID_WID)
    quadrant_j = to_quads(xyz0[:,1], GRID_WID)
    keep = np.zeros(len(feat_dist), dtype=bool)
    num_remaining_quads = GRID_WID**2
    num_remaining_samples = TOTAL_NUM          
    for qi in range(GRID_WID):
        for qj in range(GRID_WID):
            samples_per_quad = int(np.ceil(num_remaining_samples / num_remaining_quads))
            is_quad_mask = (quadrant_i == qi) & (quadrant_j == qj)  
            is_quad_inds = is_quad_mask.nonzero(as_tuple=True)[0]
            if len(is_quad_inds) > samples_per_quad:
                ord = torch.argsort(feat_dist[is_quad_mask])
                is_quad_inds = is_quad_inds[ord[:samples_per_quad]]
            keep[is_quad_inds] = True
            num_remaining_samples -= len(is_quad_inds)
            num_remaining_quads -= 1

    if not(args.spatial):
        num_to_keep = keep.sum()
        ord = torch.argsort(feat_dist.cpu())
        keep = np.zeros(len(feat_dist), dtype=bool)
        keep[ord[:num_to_keep]] = True

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

        # 2. Filter correspondences:
        if args.mode == "DFR":
            corres_idx0, corres_idx1, corres_idx0_orig, corres_idx1_orig = filter_pairs_by_distance_in_feature_space(fcgf_feats0, fcgf_feats1, corres_idx0, corres_idx1, xyz0, args)                        
        elif args.mode == "MFR":
            corres_idx0_orig, corres_idx1_orig = corres_idx0, corres_idx1
            corres_idx0, corres_idx1 = nn_to_mutual(fcgf_feats0, fcgf_feats1, corres_idx0, corres_idx1)
        elif args.mode == "BFR":
            corres_idx0, corres_idx1, corres_idx0_orig, corres_idx1_orig = filter_pairs_BFR(fcgf_feats0, fcgf_feats1, corres_idx0, corres_idx1, xyz0, args)
        elif args.mode == "no_filter":
            corres_idx0_orig, corres_idx1_orig = corres_idx0, corres_idx1
        else:
            assert False, "when running with algo==RANSAC, must define mode to either DFR or MFR"

        filter_time = time() - start_time

        num_pairs_filtered = len(corres_idx0)
        inlier_ratio_filtered = measure_inlier_ratio(corres_idx0, corres_idx1, pcd0, pcd1, T_gt, voxel_size)

    start_time = time()

    ransac_iters = 500*10**3
    if args.iters is not None:
        ransac_iters = args.iters

    # 3. Perform RANSAC
    if args.algo == "GC":        
        
        A = xyz0_np[corres_idx0,:].astype(np.float32)
        B = xyz1_np[corres_idx1,:].astype(np.float32)
        if args.prosac:
            feat_dist = calc_distances_in_feature_space(fcgf_feats0, fcgf_feats1, corres_idx0, corres_idx1, args).detach().cpu().numpy()
            match_quality = -feat_dist
        else:
            match_quality = None

        T, GC_time = GC_RANSAC( A,B, 
                                distance_threshold=2*voxel_size,
                                num_iterations=ransac_iters,
                                args=args, match_quality=match_quality)

        
    elif args.algo == "RANSAC":
        T = RANSAC_registration(pcd0,
                                pcd1,
                                corres_idx0,
                                corres_idx1,
                                2 * voxel_size,
                                num_iterations=ransac_iters,
                                args=args)
    else:
        assert False, "unexpected algo"

    # 4. estimate motion using all inlier pairs:
    if args.algo != "GC": # GC-RANSAC has this built-in, no need to repeat it here
        corres_idx0_ = corres_idx0_orig.detach().cpu().numpy()
        corres_idx1_ = corres_idx1_orig.detach().cpu().numpy()
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
                        distance_threshold, num_iterations, args):        

    corres = np.stack((idx0, idx1), axis=1)
    corres = o3d.utility.Vector2iVector(corres)
    
    if args.ELC: # use edge-length constraints:
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
    else:
        result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
            pcd0, 
            pcd1,
            corres, 
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            max_correspondence_distance=distance_threshold, 
            ransac_n=4,         
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration=num_iterations, confidence=0.9999)
        )

    return result.transformation

