import numpy as np
import matplotlib.pyplot as plt
#import cv2
import pygcransac
from time import time

correspondences = np.loadtxt('/home/ad/old_drive/home/ad/PycharmProjects/reference/graph-cut-ransac-2/examples/../build/data/kitchen/kitchen_points.txt')
gt_pose = np.loadtxt('/home/ad/old_drive/home/ad/PycharmProjects/reference/graph-cut-ransac-2/examples/../build/data/kitchen/kitchen_gt.txt')

print("Number of correspondences loaded = ", str(len(correspondences)))

def verify_pygcransac(corrs):    
    n = len(corrs)
    points1 = np.float32([corrs[i][0:3] for i in np.arange(n)]).reshape(-1,3)
    points2 = np.float32([corrs[i][3:6] for i in np.arange(n)]).reshape(-1,3)
    
    threshold = 100.0
    pose, mask = pygcransac.findRigidTransform(
        np.ascontiguousarray(points1), 
        np.ascontiguousarray(points2),
        threshold, 0.999,
        use_sprt=True)    
    return pose, mask

def tranform_points(corrs, T):
    n = len(corrs)
    points1 = np.float32([corrs[i][0:3] for i in np.arange(n)]).reshape(-1,3)
    points2 = np.float32([corrs[i][3:6] for i in np.arange(n)]).reshape(-1,3)
    
    transformed_corrs = np.zeros((corrs.shape[0], 6))

    for i in range(n):
        p1 = np.append(correspondences[i][:3], 1)
        p2 = p1.dot(T)
        transformed_corrs[i][:3] = p2[:3]
        transformed_corrs[i][3:] = corrs[i][3:]
    return transformed_corrs
    

def calculate_error(gt_pose, est_pose):
    
    R2R1 = np.dot(gt_pose[:3, :3].T, est_pose[:3, :3])
    cos_angle = max(-1.0, min(1.0, 0.5 * (R2R1.trace() - 1.0)))
    
    err_R = np.arccos(cos_angle) * 180.0 / np.pi
    err_t = np.linalg.norm(gt_pose[3, :3] - est_pose[3, :3])
    
    return err_R, err_t


initial_T = gt_pose[:4, :]
ground_truth_T = gt_pose[4:, :]

transformed_corrs = tranform_points(correspondences, initial_T)

t = time()

gc_T, gc_mask = verify_pygcransac(transformed_corrs)
print (time()-t, ' sec gc-ransac')

gc_T = np.dot(initial_T, gc_T)
    
err_R, err_t = calculate_error(ground_truth_T, gc_T)

print ('Rotation error = ', err_R, '°')
print ('Translation error = ', err_t, ' mm')