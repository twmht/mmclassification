from mmcls.apis import single_gpu_test
from mmcls.datasets import build_dataloader, build_dataset
from mmcls.models import build_classifier
from mmcv import Config, DictAction
from mmcv.runner import load_checkpoint, load_state_dict
from tqdm import tqdm
import argparse
import os
import torch
import torch2trt

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--checkpoint', default=None, help='checkpoint to load from')
    parser.add_argument('--test', dest='test', action='store_true')

    args = parser.parse_args()

    return args


args = parse_args()
cfg = Config.fromfile(args.config)

assert(cfg.tensorrt)
assert(cfg.work_dir)
if not os.path.exists(cfg.work_dir):
    os.makedirs(cfg.work_dir)

model = build_classifier(cfg.model)
datasets = [build_dataset(test) for test in cfg.data.test]
data_loaders = [
    build_dataloader(
        ds,
        cfg.data.samples_per_gpu,
        cfg.data.workers_per_gpu,
        num_gpus=1,
        dist=False,
        shuffle=False,
        round_up=False
        ) for ds in datasets
]

if args.checkpoint:
    load_checkpoint(model, args.checkpoint)

model.eval()
model.cuda()
# check the accuracy
if args.test:
    with torch.no_grad():
        for loader in data_loaders:
            results = single_gpu_test(model, loader, return_loss=False)
            results = loader.dataset.evaluate(results)
            print (results)

model.forward = model.forward_dummy
inputs = torch.randn((1, 3, cfg.tensorrt.input_size[0], cfg.tensorrt.input_size[1])).cuda()
model = torch2trt.torch2trt(model, [inputs], max_batch_size=cfg.tensorrt.max_batch_size, fp16_mode=cfg.tensorrt.fp16_mode)
torch.save(model.state_dict(), os.path.join(cfg.work_dir, cfg.tensorrt.save_name))
