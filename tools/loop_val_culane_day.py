import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import cv2
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import argparse
import numpy as np
import random
from psic.utils.config import Config
from psic.engine.runner import Runner
from psic.datasets import build_dataloader

def main():
    args = parse_args()
    # 修改
    args.work_dirs = './work_dirs/test/Culane/dla34_onlyday'
    result_dir = 'psic/dla34/culane_onlyday/20231019_013948_lr_6e-04_b_24' 
    args.pthpath = os.path.join('./work_dirs',result_dir,'ckpt')
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(
        str(gpu) for gpu in args.gpus)

    cfg = Config.fromfile(args.config)
    cfg.gpus = len(args.gpus)

    cfg.load_from = args.load_from
    cfg.resume_from = args.resume_from
    cfg.finetune_from = args.finetune_from
    cfg.view = True
    cfg.seed = args.seed

    cfg.work_dirs = args.work_dirs if args.work_dirs else cfg.work_dirs

    cudnn.benchmark = True

    

    runner = Runner(cfg)
    # 循环验证   # 5-14 
    for i in range(5,15, 1):
        pth_file = os.path.join(args.pthpath, str(i)+'.pth')
        runner.loop_val(pth_file)
    # 单次    
    # pth_file = os.path.join(args.pthpath, str(69)+'.pth')
    # runner.loop_val(pth_file)
        


def parse_args():
    # 修改
    config_root_culane = '/opt/data/private/shduan/PSIC/configs/psic/dla34_culane_only_day.py'
    parser = argparse.ArgumentParser(description='Train a detector')
    
    parser.add_argument('--config',type=str, default=config_root_culane, help='train config file path')
    parser.add_argument('--work_dirs',
                        type=str,
                        default=None,
                        help='work dirs')
    parser.add_argument('--load_from',
                        default=None,
                        help='the checkpoint file to load from')
    parser.add_argument('--resume_from',
            default = None,
            help='the checkpoint file to resume from')
    parser.add_argument('--finetune_from',
            default=None,
            help='the checkpoint file to resume from')
    parser.add_argument('--view', action='store_true', help='whether to view')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--test',
        action='store_true',
        help='whether to test the checkpoint on testing set')
    parser.add_argument('--gpus', nargs='+', type=str, default='3')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--pthpath',
                        type=str,
                        default=None,
                        help='pth path')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()
