import numpy as np
import open3d as o3d
import os
import pandas as pd
from dataloader.paths import ApolloSouthbay_dir, balanced_sets_base_dir

ORIGINAL_DATASET_PATH = ApolloSouthbay_dir
BALANCED_SETS_PATH = balanced_sets_base_dir

class Apollo_utils():

    @staticmethod
    def dataset_directory():
        return ORIGINAL_DATASET_PATH

    @staticmethod
    def get_all_session_paths():
        session_paths_file = Apollo_utils.dataset_directory() + 'session_paths.txt'
        with open(session_paths_file, "r") as fid:
            session_paths_relative = fid.read().splitlines()

        session_paths = [Apollo_utils.dataset_directory() + p for p in session_paths_relative]
        return session_paths

    @staticmethod
    def get_session_path(session_ind):
        session_paths = Apollo_utils.get_all_session_paths()
        return session_paths[session_ind]

    @staticmethod
    def load_PC(session_ind, index): 
        session_path = Apollo_utils.get_session_path(session_ind)
        filename = session_path + "pcds/%d.pcd" % index
        assert os.path.isfile(filename), "Error: could not find file " + filename
        pcd = o3d.io.read_point_cloud(filename)
        return np.asarray(pcd.points)

    
class ApolloSouthbay_balanced:
    def __init__(self, phase):
        assert phase in ['train', 'validation', 'test']
        self.name = 'ApolloSouthbay'
        self.time_step = 0.1 # seconds between consecutive frames
        self.phase = phase
        self.U = Apollo_utils()                
        pairs_file = BALANCED_SETS_PATH + '/' + self.name + '/' + phase + '.txt'
        self.pairs = pd.read_csv(pairs_file, sep=" ", header=0).values                

    def get_pair(self, ind):        
        pair = self.pairs[ind]        
        session_ind = int(pair[0])
        src_ind = int(pair[1])
        tgt_ind = int(pair[2])
        mot = pair[3:(3+16)].reshape([4,4])
        self.U.init_session(session_ind)
        A = self.U.load_PC(session_ind, src_ind)
        B = self.U.load_PC(session_ind, tgt_ind)
        return mot, A, B
