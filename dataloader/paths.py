import os 

computer_name = os.uname()[1]

if computer_name in ['ad-2021']:
  kitti_dir = '/home/ad/old_drive/data/kitti/dataset/'
  fcgf_weights_file = '/home/ad/old_drive/data/FCGF/ResUNetBN2C-feat32-kitti-v0.3.pth'
  ApolloSouthbay_dir = '/home/ad/old_drive/data/apollo_subsets/apollo_meta/'
  balanced_sets_base_dir = '/home/ad/old_drive/home/ad/PycharmProjects/BalancedDatasetGenerator/output/balanced_sets/'
  LyftLEVEL5_dir =  '/home/ad/old_drive/data/LyftLEVEL5/'
  NuScenes_dir =  '/home/ad/old_drive/data/NuScenes/'
  cache_dir =  '/home/ad/old_drive/data/balanced_sets_cache/'
elif computer_name in ['deep3d']:
  kitti_dir = '/data/amnon/datasets/kitti/dataset/'
  fcgf_weights_file = '/data/amnon/FCGF/ResUNetBN2C-feat32-kitti-v0.3.pth'
  ApolloSouthbay_dir = '/data/amnon/datasets/apollo/'  
  balanced_sets_base_dir = '/home/amnon/BalancedDatasetGenerator/output/balanced_sets/'  
  LyftLEVEL5_dir =  '/data/amnon/datasets/LyftLEVEL5/'
  NuScenes_dir =  '/data/amnon/datasets/NuScenes/'
  cache_dir =  '/data/amnon/datasets/balanced_sets_cache/'
