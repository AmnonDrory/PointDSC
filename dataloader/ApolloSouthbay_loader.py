# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu) and Wei Dong (weidong@andrew.cmu.edu)
#
# Please cite the following papers if you use any part of the code.
# - Christopher Choy, Wei Dong, Vladlen Koltun, Deep Global Registration, CVPR 2020
# - Christopher Choy, Jaesik Park, Vladlen Koltun, Fully Convolutional Geometric Features, ICCV 2019
# - Christopher Choy, JunYoung Gwak, Silvio Savarese, 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks, CVPR 2019
import os
import glob
import pickle
import pandas as pd

from dataloader.base_loader import *
from dataloader.transforms import *
from util.pointcloud import get_matching_indices, make_open3d_point_cloud

computer_name = os.uname()[1]
if computer_name in ['ad-2021']:
  apollo_southbay_dir = '/home/ad/old_drive/data/apollo_subsets/apollo_meta/'
  balanced_sets_base_dir = '/home/ad/old_drive/home/ad/PycharmProjects/BalancedDatasetGenerator/output/balanced_sets/'
elif computer_name in ['deep3d']:
  apollo_southbay_dir = '/data/amnon/datasets/apollo/'  
  balanced_sets_base_dir = '/home/amnon/BalancedDatasetGenerator/output/balanced_sets/'
else:
    assert False


class Apollo_FCGF_utils():

    @staticmethod
    def init_session(session_ind):
        pass

    @staticmethod
    def is_train_session(session_ind):
        session_path = Apollo_FCGF_utils.get_session_path(session_ind)
        return ("TrainData" in session_path)

    @staticmethod
    def is_test_session(session_ind):
        session_path = Apollo_FCGF_utils.get_session_path(session_ind)
        return ("Test" in session_path)

    def is_validation_session(session_ind):
        return Apollo_FCGF_utils.is_test_session(session_ind)
    
    @staticmethod
    def is_phase_session(session_ind, phase):
        if (phase == 'train'):
            return Apollo_FCGF_utils.is_train_session(session_ind)
        elif  (phase == 'val'):
            return Apollo_FCGF_utils.is_validation_session(session_ind)
        elif  (phase == 'test'):
            return Apollo_FCGF_utils.is_test_session(session_ind)            
        else:
            assert False, "unknown phase: " + phase

    @staticmethod
    def dataset_directory():
        return apollo_southbay_dir

    @staticmethod
    def get_all_session_paths():
        session_paths_file = Apollo_FCGF_utils.dataset_directory() + 'session_paths.txt'
        with open(session_paths_file, "r") as fid:
            session_paths_relative = fid.read().splitlines()

        session_paths = [Apollo_FCGF_utils.dataset_directory() + p for p in session_paths_relative]
        return session_paths

    @staticmethod
    def get_session_path(session_ind):
        session_paths = Apollo_FCGF_utils.get_all_session_paths()
        return session_paths[session_ind]

    @staticmethod
    def num_sessions():
        session_paths = Apollo_FCGF_utils.get_all_session_paths()
        return len(session_paths)

    @staticmethod
    def load_PC(session_ind, index):
        session_path = Apollo_FCGF_utils.get_session_path(session_ind)
        filename = session_path + "pcds/%d.pcd" % index
        if not os.path.isfile(filename):
            source = filename.replace(apollo_southbay_dir,'amnon@Geoffrey:/mnt3/dataset/apollo/')
            cmd = 'rsync -avzh %s %s' % (source, filename)
            print(cmd)
        assert os.path.isfile(filename), "Error: could not find file " + filename
        pcd = o3d.io.read_point_cloud(filename)
        return np.asarray(pcd.points).astype(np.float32)

    @staticmethod
    def get_pairs(session_ind):
        ses_path = Apollo_FCGF_utils.get_session_path(session_ind)
        pairs_file = ses_path + 'pairs/FCGF_poses_gt_for_DGR.txt'
        with open(pairs_file, 'r') as fid:
            lines = fid.read().splitlines()
        pairs = []
        for line in lines:
            pairs.append([session_ind] + [float(a) for a in line.split()])

        return pairs

class ApolloSouthbayPairDataset(PairDataset):
  AUGMENT = None
  DATA_FILES = {
      'train': './dataloader/split/train_kitti.txt',
      'val': './dataloader/split/val_kitti.txt',
      'test': './dataloader/split/test_kitti.txt'
  }
  TEST_RANDOM_ROTATION = False
  IS_ODOMETRY = True

  def prep_pairs_list(self):
      if self.SUBSET == 'balanced':
          BALANCED_SETS_DIR = balanced_sets_base_dir + 'ApolloSouthbay/'
          pairs_file = BALANCED_SETS_DIR + self.phase.replace('val','validation') + '.txt'
          pairs = pd.read_csv(pairs_file, sep=" ", header=0).values
          pairs_GT = pairs[:,:3+16]
          return pairs_GT

      pairs_GT = []
      for session_ind in range(self.U.num_sessions()):
          if not self.U.is_phase_session(session_ind, self.phase):
              continue
          self.U.init_session(session_ind)

          cur_pairs_GT = self.U.get_pairs(session_ind)

          pairs_GT.append(cur_pairs_GT)

      pairs_GT = np.vstack(pairs_GT)

      if self.phase in ['val', 'test']:
          NUM_VALIDATION_SAMPLES = 210
          N = pairs_GT.shape[0]            
          validation_inds = np.round(np.linspace(0, N-1, NUM_VALIDATION_SAMPLES)).astype(int)
          validation_mask = np.zeros([N], dtype=bool)
          test_mask = np.ones([N], dtype=bool)
          validation_mask[validation_inds] = True
          test_mask[validation_inds] = False
          if self.phase == 'val':
              pairs_GT = pairs_GT[validation_mask,:]
          elif self.phase == 'test':
              pairs_GT = pairs_GT[test_mask,:]

      return pairs_GT

  def __init__(self,
               phase,
               transform=None,
               random_rotation=True,
               random_scale=True,
               manual_seed=False,
               config=None,
               rank=None):
    # For evaluation, use the odometry dataset training following the 3DFeat eval method    
    random_rotation = self.TEST_RANDOM_ROTATION    
    PairDataset.__init__(self, phase, transform, random_rotation, random_scale,
                         manual_seed, config, rank)

    if self.rank == 0:
      logging.info(f"Loading the subset {phase}")
    
    self.phase = phase
    self.U = Apollo_FCGF_utils()
    self.pairs = self.prep_pairs_list()

  def __len__(self):
    return len(self.pairs)

  def __getitem__(self, idx):
    pair = self.pairs[idx]
    drive = int(pair[0])
    t0, t1 = int(pair[1]), int(pair[2])
    M2 = pair[3:].reshape([4,4])

    xyz0 = self.U.load_PC(drive, t0)
    xyz1 = self.U.load_PC(drive, t1)

    if self.random_rotation:
      T0 = sample_almost_planar_rotation(self.randg)
      T1 = sample_almost_planar_rotation(self.randg)
      trans = T1 @ M2 @ np.linalg.inv(T0)

      xyz0 = self.apply_transform(xyz0, T0)
      xyz1 = self.apply_transform(xyz1, T1)
    else:
      trans = M2

    matching_search_voxel_size = self.matching_search_voxel_size
    if self.random_scale and random.random() < 0.95:
      scale = self.min_scale + \
          (self.max_scale - self.min_scale) * random.random()
      matching_search_voxel_size *= scale
      xyz0 = scale * xyz0
      xyz1 = scale * xyz1
      M2[:3,3] *= scale # this line should appear here, as well as in load_kitti, in FCGF code and anywhere scaling occurs. Its absence is a bug.

    # Voxelization
    xyz0_th = torch.from_numpy(xyz0)
    xyz1_th = torch.from_numpy(xyz1)

    _, sel0 = ME.utils.sparse_quantize(xyz0_th / self.voxel_size, return_index=True)
    _, sel1 = ME.utils.sparse_quantize(xyz1_th / self.voxel_size, return_index=True)

    # Make point clouds using voxelized points
    pcd0 = make_open3d_point_cloud(xyz0[sel0])
    pcd1 = make_open3d_point_cloud(xyz1[sel1])

    # Get matches
    matches = get_matching_indices(pcd0, pcd1, trans, matching_search_voxel_size)

    # Get features
    npts0 = len(sel0)
    npts1 = len(sel1)

    feats_train0, feats_train1 = [], []

    unique_xyz0_th = xyz0_th[sel0]
    unique_xyz1_th = xyz1_th[sel1]

    feats_train0.append(torch.ones((npts0, 1)))
    feats_train1.append(torch.ones((npts1, 1)))

    feats0 = torch.cat(feats_train0, 1)
    feats1 = torch.cat(feats_train1, 1)

    coords0 = torch.floor(unique_xyz0_th / self.voxel_size)
    coords1 = torch.floor(unique_xyz1_th / self.voxel_size)

    if self.transform:
      coords0, feats0 = self.transform(coords0, feats0)
      coords1, feats1 = self.transform(coords1, feats1)

    extra_package = {'drive': drive, 't0': t0, 't1': t1}

    return (unique_xyz0_th.float(),
            unique_xyz1_th.float(), coords0.int(), coords1.int(), feats0.float(),
            feats1.float(), matches, trans, extra_package)

class ApolloSouthbayNMPairDataset(ApolloSouthbayPairDataset):
  SUBSET='10m'

class ApolloSouthbayBalancedPairDataset(ApolloSouthbayPairDataset):
  SUBSET='balanced'  

if __name__ == "__main__":
  with open('config.pickle', 'rb') as fid:
        config = pickle.load(fid)
  
  AS = ApolloSouthbayNMPairDataset('train', rank=0, config=config)
  r = AS.__getitem__(0)
  print(r)