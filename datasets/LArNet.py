import os
import torch
import numpy as np
import torch.utils.data as data
from .io import IO
from .build import DATASETS
import lmdb
import io
from utils.misc import fps
from utils.logger import *

@DATASETS.register_module()
class LArNet(data.Dataset):
    def __init__(self, config):
        self.lmdb_dir = config.LMDB_DIR
        self.num_shards = config.NUM_SHARDS
        self.npoints = config.N_POINTS
        print_log(f'[DATASET] sample out {self.npoints} points', logger = 'LArNet')

        # Initialize LMDB environments
        self.lmdb_envs = []
        for i in range(self.num_shards):
            print_log(f'[DATASET] Open shard {i}', logger = 'LArNet')
            lmdb_path = os.path.join(self.lmdb_dir, f'shard_{i}.lmdb')
            env = lmdb.open(
                lmdb_path,
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
            self.lmdb_envs.append(env)

        print_log(f'[DATASET] {len(self.lmdb_envs)} shards were loaded', logger = 'LArNet')

        # Build index mapping
        self.keys = []
        self.shard_indices = []
        print_log(f'[DATASET] Building index', logger = 'LArNet')
        self._build_index()
        print_log(f'[DATASET] {len(self.keys)} instances were loaded', logger = 'LArNet')

    def _build_index(self):
        # Iterate over each shard and collect keys
        for shard_idx, env in enumerate(self.lmdb_envs):
            with env.begin(write=False) as txn:
                cursor = txn.cursor()
                for key, _ in cursor:
                    self.keys.append(key)
                    self.shard_indices.append(shard_idx)

        self.length = len(self.keys)

    def __len__(self):
        return self.length
    
    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        # centroid = np.mean(pc, axis=0)
        centroid = torch.tensor([760.0, 760.0, 760.0]) / 2
        pc[:, :3] = pc[:, :3] - centroid
        # m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        m = 760.0 / 2
        pc[:, :3] = pc[:, :3] / m
        return pc
        
    def random_sample(self, pc, num):
        if num == -1:
            return pc
        # np.random.shuffle(self.permutation)
        # pc = pc[self.permutation[:num]]
        return fps(pc, num)


    def __getitem__(self, idx):
        key = self.keys[idx]
        shard_idx = self.shard_indices[idx]
        env = self.lmdb_envs[shard_idx]

        # Read from LMDB
        with env.begin(write=False) as txn:
            serialized_pc = txn.get(key)

        # Deserialize
        buffer = io.BytesIO(serialized_pc)
        data = np.load(buffer)  # Shape: (N, 4)

        # Downsample, norz
        data = torch.from_numpy(data).float()
        data = self.random_sample(data, self.npoints)
        data = self.pc_norm(data)

        return 'LArNet', 'sample', data

    def __del__(self):
        # Close LMDB environments
        for env in self.lmdb_envs:
            env.close()