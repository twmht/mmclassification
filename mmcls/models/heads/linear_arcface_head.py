import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import HEADS
from .cls_head import ClsHead

@HEADS.register_module()
class LinearArcFaceHead(ClsHead):
    def __init__(self,
            num_classes,
            in_channels,
            margin,
            scale,
            t,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            topk=(1, )
            ):
        super().__init__(loss=loss, topk=topk)
        self.weight = nn.Parameter(torch.empty(in_channels, num_classes))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.margin = margin
        self.s = scale
        self.t = t

    def forward_train(self, x, label):
        if not self.training:
            kernel_norm = F.normalize(self.weight, dim=0)
            x = F.normalize(x, dim= 1)
            cosine = torch.mm(x, kernel_norm)
            return cosine

        if self.t < 1:
            logits = self.hard_mining(x, label)
        else:
            kernel_norm = F.normalize(self.weight, dim=0)
            x = F.normalize(x, dim= 1)
            cosine = torch.mm(x, kernel_norm)
            theta = torch.acos(cosine)
            m = torch.zeros_like(theta)
            m.scatter_(1, label.view(-1, 1), self.margin)
            logits = self.s * (math.pi - 2 * (theta + m)) / math.pi

        losses = self.loss(logits, label)
        return losses

    def simple_test(self, x):
        return F.normalize(x, dim=1)

    def hard_mining(self, x, label):
        kernel_norm = F.normalize(self.weight, dim=0)
        x = F.normalize(x, dim= 1)
        cosine = torch.mm(x, kernel_norm)
        theta = torch.acos(cosine)
        batch_size = label.size(0)
        gt = theta[torch.arange(0, batch_size), label.view(-1)].view(-1, 1)
        mask = theta < (gt + self.margin)
        hard_example = theta[mask]
        theta[mask] = self.t * hard_example
        theta.scatter_(1, label.view(-1, 1), gt + self.margin)
        logits = self.s * (math.pi - 2 * theta) / math.pi
        return logits
