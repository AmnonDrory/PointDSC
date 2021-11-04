import numpy as np
import torch
from copy import deepcopy
from scipy.spatial.distance import cdist

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
    sixes = np.hstack([np.array(pcd0.points)[corres_idx0,:], np.array(pcd1.points)[corres_idx1,:]])
    inlier_sixes = sixes[is_close,:]
    n,_ = inlier_sixes.shape
    six_dist = cdist(inlier_sixes,inlier_sixes)
    s = np.sort(six_dist,axis=1)
    med = np.median(s,axis=0)
    m = np.min(s,axis=0)
    M = np.max(s,axis=0)


    return float(is_close.sum()) / len(is_close)
