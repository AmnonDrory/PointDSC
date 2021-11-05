import numpy as np
import torch
from copy import deepcopy

def find_nn(F0, F1, return_2nd=False):
    nn_max_n = 250

    def knn_dist(f0, f1):
        # Fast implementation with torch.einsum()
        with torch.no_grad():      
            # L2 distance:
            dist2 = torch.sum(f0**2, dim=1).reshape([-1,1]) + torch.sum(f1**2, dim=1).reshape([1,-1]) -2*torch.einsum('ac,bc->ab', f0, f1)
            dist = dist2.clamp_min(1e-30).sqrt_()
            # Cosine distance:
            #   dist = 1-torch.einsum('ac,bc->ab', f0, f1)                  

            _, ind = dist.min(dim=1, keepdim=True)                
            if return_2nd: 
                dist[torch.arange(len(ind)), ind.squeeze()] = np.inf
                _, ind_2nd = dist.min(dim=1, keepdim=True)
                
                return ind, ind_2nd
            else:
                return ind, None
    
    N = len(F0)
    C = int(np.ceil(N / nn_max_n))
    stride = nn_max_n
    inds = []
    inds_2nd = []
    for i in range(C):
        with torch.no_grad():
            ind, ind_2nd = knn_dist(F0[i * stride:(i + 1) * stride], F1)
            inds.append(ind)
            inds_2nd.append(ind_2nd)
    
    inds = torch.cat(inds)
    if return_2nd:
        inds_2nd = torch.cat(inds_2nd)
    assert len(inds) == N

    corres_idx0 = torch.arange(len(inds)).long().squeeze()
    corres_idx1 = inds.long().squeeze().cpu()
    if return_2nd:
        idx1_2nd = inds_2nd.long().squeeze().cpu()
        return corres_idx0, corres_idx1, idx1_2nd
    else:
        return corres_idx0, corres_idx1, None

def torch_intersect(Na, Nb, i_ab,j_ab,i_ba,j_ba):    
    def make_sparse_mat(i_,j_,sz):
        inds = torch.cat([i_.reshape([1,-1]),
                            j_.reshape([1,-1])],dim=0)
        vals = torch.ones_like(inds[0,:])
        
        M = torch.sparse.FloatTensor(inds,vals,sz)
        return M

    sz = [Na,Nb]
    M_ab = make_sparse_mat(i_ab,j_ab,sz)
    M_ba = make_sparse_mat(i_ba,j_ba,sz)

    M = M_ab.add(M_ba).coalesce()
    i, j = M._indices()
    v = M._values()
    is_both = (v == 2)
    i_final = i[is_both]
    j_final = j[is_both]

    return i_final, j_final

def calc_distance_ratio_in_feature_space(fcgf_feats0, fcgf_feats1, corres_idx0, corres_idx1, idx1_2nd):
    # calculate the ratio between distance to 1st and idstance to 2nd nearest neighbor, in feature space
    eps = 10**-6
    A = fcgf_feats0[corres_idx0,:]
    B_1 = fcgf_feats1[corres_idx1,:]
    B_2 = fcgf_feats1[idx1_2nd,:]
    dist_1 = torch.sqrt(torch.sum((A-B_1)**2,axis=1))        
    dist_2 = torch.sqrt(torch.sum((A-B_2)**2,axis=1))        
    feat_dist = dist_1 / (dist_2+eps)
    return feat_dist

def Grid_Prioritized_Filter(fcgf_feats0, fcgf_feats1, corres_idx0, corres_idx1, idx1_2nd, xyz0, args):
    # cobination of DFR and MFR.
    # first step - take the mutual-nearest-neighbors as a core
    # second step - use a spatial grid to add more pairs, so that every cell has some representatives.
                    # Seelct additional representatives by distances in feature space 
    
    
    is_bb, num_bb = mark_best_buddies(fcgf_feats0, fcgf_feats1, corres_idx0, corres_idx1)
    
    GRID_WID = args.BFR_grid_wid
    TOTAL_NUM = args.BFR_factor*num_bb
    if args.BFR_strict and (TOTAL_NUM < num_bb):
        TOTAL_NUM = num_bb

    feat_dist = calc_distance_ratio_in_feature_space(fcgf_feats0, fcgf_feats1, corres_idx0, corres_idx1, idx1_2nd)
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
    idx1_2nd_orig = deepcopy(idx1_2nd)
    corres_idx0 = corres_idx0[keep]
    corres_idx1 = corres_idx1[keep]
    norm_feat_dist = norm_feat_dist[keep]
    if idx1_2nd is not None:
        idx1_2nd = idx1_2nd[keep]
    else:
        idx1_2nd = None

    return corres_idx0, corres_idx1, idx1_2nd, corres_idx0_orig, corres_idx1_orig, idx1_2nd_orig, norm_feat_dist

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

def nn_to_mutual(feats0, feats1, corres_idx0, corres_idx1, idx1_2nd=None, force_return_2nd=False):
    
    uniq_inds_1=torch.unique(corres_idx1)
    inv_corres_idx1, inv_corres_idx0, _ = find_nn(feats1[uniq_inds_1,:], feats0, False)
    inv_corres_idx1 = uniq_inds_1

    final_corres_idx0, final_corres_idx1 = torch_intersect(
    feats0.shape[0], feats1.shape[0],
    corres_idx0, corres_idx1,
    inv_corres_idx0, inv_corres_idx1)

    if idx1_2nd is not None:
        idx1_2nd = idx1_2nd[final_corres_idx0] # this relies on the fact that corres_idx0 is the full sorted range [0,...,n]
        return final_corres_idx0, final_corres_idx1, idx1_2nd
    elif force_return_2nd:
        return final_corres_idx0, final_corres_idx1, None
    else:
        return final_corres_idx0, final_corres_idx1

def measure_inlier_ratio(corres_idx0, corres_idx1, pcd0, pcd1, T_gt, voxel_size):
    corres_idx0_ = corres_idx0.detach().numpy()
    corres_idx1_ = corres_idx1.detach().numpy()
    pcd0_trans = deepcopy(pcd0)
    pcd0_trans.transform(T_gt)
    
    dist2 = np.sum((np.array(pcd0_trans.points)[corres_idx0_,:] - np.array(pcd1.points)[corres_idx1_,:])**2, axis=1)
    is_close = dist2 < (2*voxel_size)**2
    return float(is_close.sum()) / len(is_close)
