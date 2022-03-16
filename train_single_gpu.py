#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import random
import warnings
from loguru import logger

import torch

# user generated imports
from yolox.core import init_trainer_by_process_name
from yolox.exp import init_exp_by_process_name

def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")
    parser.add_argument("-expn", "--experiment-name", type=str)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("-cfg", "--config", type=str, default="comic", 
                        help="type of the config file: [comic, boost, styled, unsupervised, ...]")
    
    parser.add_argument("-ms", "--model-size", type=str, default="large", 
                        help="model size: [xs, s, m, l, xl]")
    parser.add_argument("-hm", "--head-mode", type=int, default=0, 
                        help="0 for both body and face, 1 for face, 2 for body")
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("--resume", default=False, action="store_true", help="resume training")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument("-e", "--start-epoch", default=None, type=int, 
                        help="resume training start epoch")
    parser.add_argument("-l", "--image-limit", default=None, type=int, 
                        help="limit the number of images seen during training, only works in fully comic supervised mode")
    
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision training.",
    )
    parser.add_argument(
        "-o",
        "--occupy",
        dest="occupy",
        default=False,
        action="store_true",
        help="occupy GPU memory first for training.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    
    args = make_parser().parse_args()
    
    assert args.experiment_name
    assert args.model_size
    assert args.head_mode in [0, 1, 2]
         
    Exp = init_exp_by_process_name(args.config)
    exp = Exp(args.model_size, args.head_mode, dataset_name=args.config)
    exp.merge(args.opts)
    if args.image_limit is not None:
        exp.image_limit = args.image_limit
    
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )

    # set environment variables for distributed training
    Trainer = init_trainer_by_process_name(args.config)
    trainer = Trainer(exp, args)
    trainer.train()