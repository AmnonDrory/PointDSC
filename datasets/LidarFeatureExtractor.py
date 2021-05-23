import os
import torch.utils.data as data
from utils.pointcloud import make_point_cloud, estimate_normal
from utils.SE3 import *
from dataloader.kitti_loader import KITTINMPairDataset # AD TODO - this doesn't belong here. The actual dset should be supplied as an argument to __init__
from easydict import EasyDict as edict # AD TODO - this doesn't belong here.
from misc.fcgf import ResUNetBN2C as FCGF
import MinkowskiEngine as ME

from general.paths import kitti_dir, fcgf_weights_file

class LidarFeatureExtractor(data.Dataset):
    def __init__(self,
                dataset,
                in_dim=6,
                num_node=5000,
                use_mutual=True,
                augment_axis=0,
                augment_rotation=1.0,
                augment_translation=0.01,
                ):

        self.in_dim = in_dim
        self.descriptor = 'fcgf'
        self.num_node = num_node
        self.use_mutual = use_mutual
        self.augment_axis = augment_axis
        self.augment_rotation = augment_rotation
        self.augment_translation = augment_translation

        # containers
        self.ids_list = []

        self.dataset = dataset
        self.split = self.dataset.phase
        self.downsample = self.dataset.voxel_size
        self.inlier_threshold = 2*self.dataset.voxel_size

        self.model = FCGF(
            1,
            32,
            bn_momentum=0.05,
            conv1_kernel_size=5,
            normalize_feature=True
        ).cuda()
        checkpoint = torch.load(fcgf_weights_file)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()               


    def __getitem__(self, index):
        
        unique_xyz0_th, unique_xyz1_th, coords0, coords1, feats0, feats1, matches, trans, extra_package = self.dataset.__getitem__(index)
        coords0, coords1, feats0, feats1 = coords0.cuda(), coords1.cuda(), feats0.cuda(), feats1.cuda()

        # AD TODO: it might be wasteful that we send each cloud separately through the FCGF network, instead of sending an entire batch together.
        coords0 = ME.utils.batched_coordinates([coords0.float()])
        stensors0 = ME.SparseTensor(feats0, coordinates=coords0, device=feats0.device)
        features0 = self.model(stensors0).F        

        coords1 = ME.utils.batched_coordinates([coords1.float()])
        stensors1 = ME.SparseTensor(feats1, coordinates=coords1, device=feats0.device)
        features1 = self.model(stensors1).F        

        src_keypts = unique_xyz0_th.detach().cpu().numpy()
        tgt_keypts = unique_xyz1_th.detach().cpu().numpy()
        src_features = features0.detach().cpu().numpy()
        tgt_features = features1.detach().cpu().numpy()
        if self.descriptor == 'fpfh':
            src_features = src_features / (np.linalg.norm(src_features, axis=1, keepdims=True) + 1e-6)
            tgt_features = tgt_features / (np.linalg.norm(tgt_features, axis=1, keepdims=True) + 1e-6)

        # compute ground truth transformation
        orig_trans = trans
        
        # AD OBSERVATION - dramatic augmentation of the xyz clouds, AFTER the
        #           FCGF features have already been calculated. This is
        #           not realistic, it gives FCGF oracle level data. 

        # data augmentation
        if self.split == 'train':
            src_keypts += np.random.rand(src_keypts.shape[0], 3) * 0.05
            tgt_keypts += np.random.rand(tgt_keypts.shape[0], 3) * 0.05
        
        # AD OBSERVATION: very dramatic rotation, not reasonable for 
        #                 automotive lidar datasets, e.g. kitti. Would be
        #                 more reasonable to limt rotation to almost-planar

        # AD OBSERVATION: this augmentation is also performed on validation set. Therefore, validation outputs are
        # not deterministic. For test setm the augmentation parameters are set to 0, therefore the
        # augmentation functions do nothing.

        aug_R = rotation_matrix(self.augment_axis, self.augment_rotation)
        aug_T = translation_matrix(self.augment_translation)
        aug_trans = integrate_trans(aug_R, aug_T)
        tgt_keypts = transform(tgt_keypts, aug_trans)
        gt_trans = concatenate(aug_trans, orig_trans)

        # select {self.num_node} numbers of keypoints
        N_src = src_features.shape[0]
        N_tgt = tgt_features.shape[0]
        src_sel_ind = np.arange(N_src)
        tgt_sel_ind = np.arange(N_tgt)
        # AD OBSERVATION: for the test set, determinism here is achieved by setting the seed
        if self.num_node != 'all' and N_src > self.num_node:
            src_sel_ind = np.random.choice(N_src, self.num_node, replace=False)
        if self.num_node != 'all' and N_tgt > self.num_node:
            tgt_sel_ind = np.random.choice(N_tgt, self.num_node, replace=False)

        # AD OBSERVATION: extreme downsampling of clouds, from ~20K to 1K. 
        src_desc = src_features[src_sel_ind, :]
        tgt_desc = tgt_features[tgt_sel_ind, :]
        src_keypts = src_keypts[src_sel_ind, :]
        tgt_keypts = tgt_keypts[tgt_sel_ind, :]

        # construct the correspondence set by mutual nn in feature space.
        # AD OBSERVATION: interesting definition of distance
        distance = np.sqrt(2 - 2 * (src_desc @ tgt_desc.T) + 1e-6)
        source_idx = np.argmin(distance, axis=1)
        if self.use_mutual:
            target_idx = np.argmin(distance, axis=0)
            mutual_nearest = (target_idx[source_idx] == np.arange(source_idx.shape[0]))
            corr = np.concatenate([np.where(mutual_nearest == 1)[0][:,None], source_idx[mutual_nearest][:,None]], axis=-1)
        else:
            corr = np.concatenate([np.arange(source_idx.shape[0])[:, None], source_idx[:, None]], axis=-1)

        # compute the ground truth label
        frag1 = src_keypts[corr[:, 0]]
        frag2 = tgt_keypts[corr[:, 1]]
        frag1_warp = transform(frag1, gt_trans)
        distance = np.sqrt(np.sum(np.power(frag1_warp - frag2, 2), axis=1))
        labels = (distance < self.inlier_threshold).astype(np.int)

        # add random outlier to input data
        if self.split == 'train' and np.mean(labels) > 0.5:
            num_outliers = int(0.0 * len(corr))
            src_outliers = np.random.randn(num_outliers, 3) * np.mean(src_keypts, axis=0)
            tgt_outliers = np.random.randn(num_outliers, 3) * np.mean(tgt_keypts, axis=0)
            input_src_keypts = np.concatenate( [src_keypts[corr[:, 0]], src_outliers], axis=0)
            input_tgt_keypts = np.concatenate( [tgt_keypts[corr[:, 1]], tgt_outliers], axis=0)
            labels = np.concatenate( [labels, np.zeros(num_outliers)], axis=0)
        else:
            # prepare input to the network
            input_src_keypts = src_keypts[corr[:, 0]]
            input_tgt_keypts = tgt_keypts[corr[:, 1]]

        if self.in_dim == 3:
            corr_pos = input_src_keypts - input_tgt_keypts
        elif self.in_dim == 6:
            corr_pos = np.concatenate([input_src_keypts, input_tgt_keypts], axis=-1)
            # move the center of each point cloud to (0,0,0).
            corr_pos = corr_pos - corr_pos.mean(0)
        elif self.in_dim == 9:
            corr_pos = np.concatenate([input_src_keypts, input_tgt_keypts, input_src_keypts-input_tgt_keypts], axis=-1)
        elif self.in_dim == 12:
            src_pcd = make_point_cloud(src_keypts)
            tgt_pcd = make_point_cloud(tgt_keypts)
            estimate_normal(src_pcd, radius=self.downsample*2)
            estimate_normal(tgt_pcd, radius=self.downsample*2)
            src_normal = np.array(src_pcd.normals)
            tgt_normal = np.array(tgt_pcd.normals)
            src_normal = src_normal[src_sel_ind, :]
            tgt_normal = tgt_normal[tgt_sel_ind, :]
            input_src_normal = src_normal[corr[:, 0]]
            input_tgt_normal = tgt_normal[corr[:, 1]]
            corr_pos = np.concatenate([input_src_keypts, input_src_normal, input_tgt_keypts, input_tgt_normal], axis=-1)

        return corr_pos.astype(np.float32), \
            input_src_keypts.astype(np.float32), \
            input_tgt_keypts.astype(np.float32), \
            gt_trans.astype(np.float32), \
            labels.astype(np.float32),

    def __len__(self):
        return len(self.dataset)

if __name__ == "__main__":
    phase = 'train'
    dset = KITTINMPairDataset(
        phase,
        transform=None,
        random_rotation=False,
        random_scale=False,
        manual_seed=False,
        config=edict(
            {'kitti_dir': os.path.split(kitti_dir)[0], 
            'icp_cache_path': 'icp',
            'voxel_size': 0.3, 
            'positive_pair_search_voxel_size_multiplier': 1.5,
            }),
        rank=0)

    ex = LidarFeatureExtractor(
                    dataset=dset,
                    num_node=5000,
                    use_mutual=False,
                    augment_axis=0,
                    augment_rotation=0,
                    augment_translation=0.00
                    )

    print(len(ex))
    for i in range(ex.__len__()):
        ret_dict = ex.__getitem__(i)
        for k in ret_dict:
            print('%f ' % k.flatten()[0], end=None)
        print('')
