import torch
import torch.nn as nn
from collections import OrderedDict

from ..builder import NECKS

def conv1x1(
    in_channels, out_channels, module_name, postfix, stride=1, groups=1, kernel_size=1, padding=0
):
    """1x1 convolution with padding"""
    return [
        (
            f"{module_name}_{postfix}/conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
        ),
        (f"{module_name}_{postfix}/norm", nn.BatchNorm2d(out_channels)),
        (f"{module_name}_{postfix}/relu", nn.PReLU(out_channels)),
    ]

@NECKS.register_module()
class GlobalConvolutionPooling(nn.Module):
    """Global Average Pooling neck.

    Note that we use `view` to remove extra channel after pooling.
    We do not use `squeeze` as it will also remove the batch dimension
    when the tensor has a batch dimension of size 1, which can lead to
    unexpected errors.
    """

    def __init__(self, input_channel, last_channel, emb_size):
        super(GlobalConvolutionPooling, self).__init__()
        self.reduce = nn.Sequential(OrderedDict(conv1x1(input_channel, last_channel, "reduce_conv", "0")))
        self.output_layer = torch.nn.Sequential(
                                       nn.Conv2d(last_channel, last_channel, kernel_size=7, groups=last_channel, bias=False),
                                       torch.nn.BatchNorm2d(last_channel),
                                       nn.Conv2d(last_channel, emb_size, kernel_size=1, bias=False),
                                       torch.nn.BatchNorm2d(emb_size)
                                       )

        self.init_weights()
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        assert(isinstance(self.output_layer[3], nn.BatchNorm2d))
        self.output_layer[3].weight.requires_grad = False

    def forward(self, inputs):
        outs = self.reduce(inputs)
        outs = self.output_layer(outs)
        outs = outs.view(inputs.size(0), -1)
        return outs
