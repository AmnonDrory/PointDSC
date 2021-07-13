import open3d as o3d
import numpy as np 
from scipy.spatial import cKDTree
from copy import deepcopy
import torch
from time import time
import teaserpp_python

def pcd2xyz(pcd):
    return np.asarray(pcd.points).T

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

def find_knn_gpu(F0, F1):
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
    return inds

def find_mutual_correspondences_GPU(feats0, feats1):
    nns = find_knn_gpu(feats0, feats1)
    corres_idx0 = torch.arange(len(nns), dtype=torch.long, device=nns.device, requires_grad=False)
    corres_idx1 = nns.long().squeeze()
    
    uniq_inds_1=torch.unique(corres_idx1)
    inv_nns = find_knn_gpu(feats1[uniq_inds_1,:], feats0)
    inv_corres_idx1 = uniq_inds_1
    inv_corres_idx0 = inv_nns.long().squeeze()

    final_corres_idx0, final_corres_idx1 = torch_intersect(
    feats0.shape[0], feats1.shape[0],
    corres_idx0, corres_idx1,
    inv_corres_idx0, inv_corres_idx1)

    return final_corres_idx0, final_corres_idx1

def get_teaser_solver(noise_bound):
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1.0
    solver_params.noise_bound = noise_bound
    solver_params.estimate_scaling = False
    solver_params.inlier_selection_mode = \
        teaserpp_python.RobustRegistrationSolver.INLIER_SELECTION_MODE.PMC_EXACT
    solver_params.rotation_tim_graph = \
        teaserpp_python.RobustRegistrationSolver.INLIER_GRAPH_FORMULATION.CHAIN
    solver_params.rotation_estimation_algorithm = \
        teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 10000
    solver_params.rotation_cost_threshold = 1e-16
    solver = teaserpp_python.RobustRegistrationSolver(solver_params)
    return solver

def Rt2T(R,t):
    T = np.identity(4)
    T[:3,:3] = R
    T[:3,3] = t
    return T 

def TEASER(A_pcd, B_pcd, A_feats, B_feats, voxel_size=0.3):
    A_xyz = pcd2xyz(A_pcd) # np array of size 3 by N
    B_xyz = pcd2xyz(B_pcd) # np array of size 3 by M
    # establish correspondences by nearest neighbour search in feature space
    device = 'cuda:%d' % torch.cuda.current_device()
    A_feats = A_feats.to(device)
    B_feats = B_feats.to(device)
    corrs_A_tensor, corrs_B_tensor = find_mutual_correspondences_GPU(A_feats, B_feats)
    corrs_A = corrs_A_tensor.cpu().detach().numpy()
    corrs_B = corrs_B_tensor.cpu().detach().numpy()
    A_corr = A_xyz[:,corrs_A] # np array of size 3 by num_corrs
    B_corr = B_xyz[:,corrs_B] # np array of size 3 by num_corrs

    # robust global registration using TEASER++
    start_time = time()
    NOISE_BOUND = voxel_size
    teaser_solver = get_teaser_solver(NOISE_BOUND)
    teaser_solver.solve(A_corr,B_corr)
    solution = teaser_solver.getSolution()
    R_teaser = solution.rotation
    t_teaser = solution.translation
    T_teaser = Rt2T(R_teaser,t_teaser)
    elapsed_time = time() - start_time

    print(f"T_teaser={T_teaser}") # AD DEL

    return T_teaser, elapsed_time