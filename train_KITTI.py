import os
import time
import shutil
import json 
from config import get_config
from easydict import EasyDict as edict
from libs.loss import TransformationLoss, ClassificationLoss, SpectralMatchingLoss
from datasets.KITTI import KITTIDataset
from datasets.dataloader import get_dataloader
from libs.trainer import Trainer
from models.PointDSC import PointDSC
from torch import optim

from dataloader.kitti_loader import KITTINMPairDataset
from datasets.LidarFeatureExtractor import LidarFeatureExtractor
from dataloader.base_loader import CollationFunctionFactory
from torch.utils.data import DataLoader

if __name__ == '__main__':
    config = get_config()
    dconfig = vars(config)

    if True: # AD UNDO
        dconfig['batch_size'] = 3 
        dconfig['num_workers'] = 1

    for k in dconfig:
        print(f"    {k}: {dconfig[k]}")
    config = edict(dconfig)
    os.makedirs(config.snapshot_dir, exist_ok=True)
    os.makedirs(config.tboard_dir, exist_ok=True)
    os.makedirs(config.save_dir, exist_ok=True)
    shutil.copy2(os.path.join('.', 'train_KITTI.py'), os.path.join(config.snapshot_dir, 'train.py'))
    shutil.copy2(os.path.join('.', 'libs/trainer.py'), os.path.join(config.snapshot_dir, 'trainer.py'))
    shutil.copy2(os.path.join('.', 'models/PointDSC.py'), os.path.join(config.snapshot_dir, 'model.py'))  # for the model setting.
    shutil.copy2(os.path.join('.', 'libs/loss.py'), os.path.join(config.snapshot_dir, 'loss.py'))
    shutil.copy2(os.path.join('.', 'datasets/KITTI.py'), os.path.join(config.snapshot_dir, 'dataset.py'))
    json.dump(
        config,
        open(os.path.join(config.snapshot_dir, 'config.json'), 'w'),
        indent=4,
    )

    # create model 
    config.model = PointDSC(
        in_dim=config.in_dim,
        num_layers=config.num_layers, 
        num_channels=config.num_channels,
        num_iterations=config.num_iterations,
        inlier_threshold=config.inlier_threshold,
        sigma_d=config.sigma_d,
        ratio=config.ratio,
        k=config.k,
    )

    # create optimizer 
    if config.optimizer == 'SGD':
        config.optimizer = optim.SGD(
            config.model.parameters(), 
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            )
    elif config.optimizer == 'ADAM':
        config.optimizer = optim.Adam(
            config.model.parameters(), 
            lr=config.lr,
            betas=(0.9, 0.999),
            # momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    config.scheduler = optim.lr_scheduler.ExponentialLR(
        config.optimizer,
        gamma=config.scheduler_gamma,
    )

    # create dataset and dataloader
    collation_fn = CollationFunctionFactory(concat_correspondences=False,
                                            collation_type='collate_pair')
    train_set = KITTINMPairDataset(
            'train',
            transform=None, random_rotation=False, random_scale=False,
            manual_seed=False, config=None, rank=0)

    val_set = KITTINMPairDataset(
            'val',
            transform=None, random_rotation=False, random_scale=False,
            manual_seed=False, config=None, rank=0)                

    config.train_loader = DataLoader(train_set,
                                    batch_size=config.batch_size,
                                    collate_fn=collation_fn,
                                    num_workers=config.num_workers,
                                    shuffle=True)

    config.val_loader = DataLoader(val_set,
                                    batch_size=config.batch_size,
                                    collate_fn=collation_fn,
                                    num_workers=config.num_workers,
                                    shuffle=False)

    config.train_feature_extractor = LidarFeatureExtractor(
            split='train',
            in_dim=config.in_dim,
            inlier_threshold=config.inlier_threshold,
            num_node=config.num_node, 
            use_mutual=config.use_mutual,
            augment_axis=config.augment_axis,
            augment_rotation=config.augment_rotation,
            augment_translation=config.augment_translation,                
            )                                        

    config.val_feature_extractor = LidarFeatureExtractor(
            split='val',
            in_dim=config.in_dim,
            inlier_threshold=config.inlier_threshold,
            num_node=config.num_node, 
            use_mutual=config.use_mutual,
            augment_axis=0,
            augment_rotation=0.0,
            augment_translation=0.0,                
            )                                        


    # create evaluation
    config.evaluate_metric = {
        "ClassificationLoss": ClassificationLoss(balanced=config.balanced),
        "SpectralMatchingLoss": SpectralMatchingLoss(balanced=config.balanced),
        "TransformationLoss": TransformationLoss(re_thre=config.re_thre, te_thre=config.te_thre),
    }
    config.metric_weight = {
        "ClassificationLoss": config.weight_classification,
        "SpectralMatchingLoss": config.weight_spectralmatching,
        "TransformationLoss": config.weight_transformation,
    }

    trainer = Trainer(config)
    trainer.train()