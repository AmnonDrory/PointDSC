import os 

computer_name = os.uname()[1]

if computer_name in ['ad-2021']:
  kitti_dir = '/home/ad/old_drive/data/kitti/dataset'
  fcgf_weights_file = '/home/ad/old_drive/data/FCGF/ResUNetBN2C-feat32-kitti-v0.3.pth'
elif computer_name in ['deep3d']:
  kitti_dir = '/data/amnon/datasets/kitti/dataset'
  fcgf_weights_file = '/data/amnon/FCGF/ResUNetBN2C-feat32-kitti-v0.3.pth'
else:
  kitti_dir = None
  fcgf_weights_file = None

from general.paths import kitti_dir, fcgf_weights_file
