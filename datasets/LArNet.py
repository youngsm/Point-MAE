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

from numba import jit

@jit(nopython=True)
def furthest_point_sampling(points, num_samples):
    """
    Perform furthest point sampling (FPS) on a set of points.
    
    Args:
        points (numpy.ndarray): The input point cloud data of shape (N, D), where N is the number of points and D is the dimensionality.
        num_samples (int): The number of points to sample.
    
    Returns:
        numpy.ndarray: The sampled points of shape (num_samples, D).
    """
    N, D = points.shape
    centroids = np.zeros((num_samples,), dtype=np.int32)
    distances = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(num_samples):
        centroids[i] = farthest
        centroid = points[farthest, :]
        dist = np.sum((points - centroid) ** 2, axis=1)
        mask = dist < distances
        distances[mask] = dist[mask]
        farthest = np.argmax(distances)
    return points[centroids]



@DATASETS.register_module()
class LArNet(data.Dataset):
    def __init__(self, config):
        self.lmdb_dir = config.LMDB_DIR
        self.npoints = config.N_POINTS
        split = config.subset
        self.subset = config.subset
        print_log(f'[DATASET] sample out {self.npoints} points', logger = 'LArNet')
        print_log(f'[DATASET] subset: {self.subset}', logger = 'LArNet')

        # Initialize LMDB environments
        lmdb_path = os.path.join(self.lmdb_dir, f'{split}.lmdb')
        self.lmdb_env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        # Build index mapping
        self.keys = []
        self.shard_indices = []
        print_log(f'[DATASET] Building index', logger = 'LArNet')
        self._build_index()
        print_log(f'[DATASET] {len(self.keys)} instances were loaded', logger = 'LArNet')

    def _build_index(self):
        # Iterate over each shard and collect keys
        with self.lmdb_env.begin(write=False) as txn:
            keys = txn.get(b'__keys__')

        # Deserialize
        buffer = io.BytesIO(keys)
        self.keys = np.load(buffer)  # Shape: (N, 4)
        self.length = len(self.keys)

    def __len__(self):
        return self.length
    
    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        # centroid = np.mean(pc, axis=0)
        centroid = np.array([760.0, 760.0, 760.0]) / 2
        pc[:, :3] = pc[:, :3] - centroid
        # m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        m = 760.0 * np.sqrt(2) / 2 
        pc[:, :3] = pc[:, :3] / m
        return pc
        
    def random_sample(self, pc, num):
        if num == -1:
            return pc
        # np.random.shuffle(self.permutation)
        # pc = pc[self.permutation[:num]]
        return furthest_point_sampling(pc, num)

    def __getitem__(self, idx):
        key = self.keys[idx]
        env = self.lmdb_env

        # Read from LMDB
        with env.begin(write=False) as txn:
            serialized_pc = txn.get(key)

        # Deserialize
        buffer = io.BytesIO(serialized_pc)
        data = np.load(buffer)  # Shape: (N, 4)

        # Downsample, normalize
        # data = torch.from_numpy(data).float()
        data = self.random_sample(data, self.npoints)
        data = self.pc_norm(data)
        data = torch.from_numpy(data).float()
        return data

    def __del__(self):
        # Close LMDB environments
        self.lmdb_env.close()