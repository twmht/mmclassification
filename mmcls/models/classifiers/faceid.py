import torch.nn as nn

from ..builder import CLASSIFIERS, build_backbone, build_head, build_neck
from .image import ImageClassifier


@CLASSIFIERS.register_module()
class FaceID(ImageClassifier):
    def forward_dummy(self, img):
        x = self.extract_feat(img)
        return x
