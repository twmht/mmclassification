import os
import os.path
import pickle
import torch

import numpy as np
import lmdb
import mmcv
from torch.utils.data import Dataset

from .base_dataset import BaseDataset
from .builder import DATASETS
from .pipelines import Compose
from mmcls.utils.verification import evaluate as face_evaluate


@DATASETS.register_module()
class FaceVerification(Dataset):

    def __init__(self, root, name, pipeline):
        db = lmdb.open(os.path.join(root, name + '.lmdb'), readonly=True)
        self.name = name
        issame = np.load(os.path.join(root, name + '_list.npy'))
        self.txn = db.begin(write=False)
        self.issame = issame
        self.num_pairs = self.issame.shape[0]
        self.pipeline = Compose(pipeline)

    def __getitem__(self, index):
        blob = self.txn.get(str(index).encode())

        #  img = Image.open(io.BytesIO(blob))
        img = mmcv.imfrombytes(blob)
        results = {'img': img}
        img = self.pipeline(results)

        return img

    def __len__(self):
        return self.num_pairs * 2

    def evaluate(self, results, metric='accuracy', logger=None):
        results = torch.cat(results, dim=0)
        assert(len(results) == len(self)), f'{len(results)}!={len(self)}'
        embedding = torch.nn.functional.normalize(results, dim=1)
        tpr, fpr, accuracy, best_thresholds = face_evaluate(embedding.cpu().numpy(), self.issame, 10)
        results = {}
        results[self.name + '/acc'] = accuracy.mean()
        return results
