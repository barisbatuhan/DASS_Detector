#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import os
import ast
import pprint
from abc import ABCMeta, abstractmethod
from typing import Dict
from tabulate import tabulate

import random

import torch
import torch.nn as nn
from torch.nn import Module

from yolox.utils import LRScheduler


class BaseExp(metaclass=ABCMeta):
    """Basic class for any experiment."""

    def __init__(self, model_size):
        
        assert model_size in ["xsmall", "xs", 
                              "small", "s", 
                              "medium", "m", 
                              "large", "l", 
                              "xlarge", "xl"]
        
        self.seed = None
        self.output_dir = "./YOLOX_outputs"
        self.print_interval = 10
        self.eval_interval  = 1
        self.num_workers    = 4
        
        # ----------------- training related unchanging parameters ----------------- #
        self.weight_decay = 5e-4
        self.momentum     = 0.9
        self.warmup_lr    = 0.001
        self.lr           = 0.001
        self.scheduler    = "yoloxwarmcos"
        self.ema          = True
        self.max_iter     = 200
        
        # ----------------- augmentation config ------------------ #
        self.enable_mixup = False
        
        # ----------------- testing config ------------------ #
        self.test_size    = (640, 640)
        self.test_conf    = 0.01
        self.nmsthre      = 0.65
        self.in_channels  = [256, 512, 1024]
        self.strides      = [8, 16, 32]
        
        if model_size in ["xsmall", "xs"]:
            self.depth = 0.33
            self.width = 0.375
        elif model_size in ["small", "s"]:
            self.depth = 0.33
            self.width = 0.50
        elif model_size in ["medium", "m"]:
            self.depth = 0.67
            self.width = 0.75
        elif model_size in ["large", "l"]:
            self.depth = 1.00
            self.width = 1.00
        elif model_size in ["xlarge", "xl"]:
            self.depth = 1.33
            self.width = 1.25
            
        userfiles_path = "/userfiles/comics_grp/datasets/"
        user_path      = "/userfiles/baristopal20/datasets/"
        cluster_path   = "/datasets/"
        icf_start      = "iCartoonFace2020/personai_icartoonface_det"
        
        self.face_data_dir = {
            "wf_train_imgs": os.path.join(userfiles_path,
                                          "widerface_styled/WIDER_train/images/"),
            "wf_test_imgs": os.path.join(userfiles_path,
                                         "widerface_styled/WIDER_val/images/"),
            "wf_train_labels": os.path.join(userfiles_path,
                                            "widerface_styled/retinaface/train/label.txt"),
            "wf_test_labels": os.path.join(userfiles_path,
                                           "widerface_styled/retinaface/val/label.txt"),
            
            "icf_train_imgs": os.path.join(cluster_path,
                                           icf_start + "train/icartoonface_dettrain/"),
            "icf_test_imgs": os.path.join(cluster_path, icf_start + "val/"),
            "icf_train_labels": os.path.join(cluster_path,
                                             icf_start + "train/icartoonface_dettrain.csv"),
            "icf_test_labels": os.path.join(userfiles_path,
                                            "icf_val_annot/personai_icartoonface_detval.csv"),
            
            "m109_frames_imgs": os.path.join(userfiles_path, "manga109_frames/imgs/"),
            "m109_frames_labels": os.path.join(userfiles_path, "manga109_frames/annots.json"),
            
            "m109": os.path.join(cluster_path, "manga109/"),
            "dcm772": os.path.join(userfiles_path, "dcm772/dcm-dataset_from_rigaud/"),

            "golden_panels": os.path.join(cluster_path, "COMICS/raw_panel_images/"),
            "golden_pages": os.path.join(cluster_path, "COMICS/raw_page_images/"),

            "dcm772_frames_imgs": os.path.join(userfiles_path, "dcm772_frames/imgs/"),
            "dcm772_frames_labels": os.path.join(userfiles_path,
                                                 "dcm772_frames/annots/annotations.json"),
            "dcm772_frames_partition": os.path.join(userfiles_path,
                                                    "dcm772_frames/annots/train.txt")
        }
        
        self.body_data_dir = {
            "m109": os.path.join(cluster_path, "manga109/"),

            "golden_panels": os.path.join(cluster_path, "COMICS/raw_panel_images/"),
            "golden_pages": os.path.join(cluster_path, "COMICS/raw_page_images/"),
            
            "m109_frames_imgs": os.path.join(userfiles_path, "manga109_frames/imgs/"),
            "m109_frames_labels": os.path.join(userfiles_path, "manga109_frames/annots.json"),
            
            "comic": os.path.join(userfiles_path, "comic2k"), 
            "watercolor": os.path.join(userfiles_path, "watercolor2k"), 
            "clipart": os.path.join(userfiles_path, "clipart2k"),
            
            "dcm772": os.path.join(userfiles_path, "dcm772/dcm-dataset_from_rigaud/"),
            "dcm772_frames_imgs": os.path.join(userfiles_path, "dcm772_frames/imgs/"),
            "dcm772_frames_labels": os.path.join(userfiles_path,
                                                 "dcm772_frames/annots/annotations.json"),
            "dcm772_frames_partition": os.path.join(userfiles_path,
                                                    "dcm772_frames/annots/train.txt"),
            
            "coco": os.path.join(userfiles_path, "COCO_styled/"),
            
            "ebd_imgs": os.path.join(userfiles_path,
                                     "eBDtheque2019/eBDtheque_database_v3/Pages/"),
            "ebd_labels": os.path.join(userfiles_path,
                                       "eBDtheque2019/eBDtheque_database_v3/GT/")
        }
    
    @abstractmethod
    def get_loss_fn(self) -> Module:
        pass
        
    def get_model(self) -> Module:
        
        from yolox.models import YOLOPAFPN, YOLOXHeadStem, YOLOX, YOLOXHead
        
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        loss_fn = self.get_loss_fn()
        
        if getattr(self, "model", None) is None:
            
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=self.in_channels)
            face_head = YOLOXHead(1, self.width, loss_fn=loss_fn, 
                                  strides=self.strides,
                                  in_channels=self.in_channels)
            body_head = YOLOXHead(1, self.width, loss_fn=loss_fn, 
                                  strides=self.strides,
                                  in_channels=self.in_channels)
            head_stem = YOLOXHeadStem(self.width, in_channels=self.in_channels)
            self.model = YOLOX(backbone, head_stem, face_head, body_head)

        self.model.apply(init_yolo)
        self.model.face_head.initialize_biases(1e-2)
        self.model.body_head.initialize_biases(1e-2)
        return self.model

    
    @abstractmethod
    def get_data_loader(
        self, batch_size: int, is_distributed: bool
    ) -> Dict[str, torch.utils.data.DataLoader]:
        pass

    def get_optimizer(self, batch_size: int) -> torch.optim.Optimizer:
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.lr
                # lr = self.basic_lr_per_img * batch_size

            pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)  # biases
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    pg0.append(v.weight)  # no decay
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg1.append(v.weight)  # apply decay

            optimizer = torch.optim.SGD(
                pg0, lr=lr, momentum=self.momentum, nesterov=True
            )
            optimizer.add_param_group(
                {"params": pg1, "weight_decay": self.weight_decay}
            )  # add pg1 with weight_decay
            optimizer.add_param_group({"params": pg2})
            self.optimizer = optimizer

        return self.optimizer


    def get_eval_loader(self, batch_size, is_distributed):

        from yolox.data import ICartoonFaceDataset, Comic2kDataset, ValTransform
        
        face_valdataset = ICartoonFaceDataset(
            data_dir=self.face_data_dir,
            train=False,
            img_size=self.test_size,
            preproc=ValTransform(legacy=False),
        )

        face_sampler = torch.utils.data.SequentialSampler(face_valdataset)
        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": False,
            "sampler": face_sampler,
        }

        dataloader_kwargs["batch_size"] = batch_size
        face_val_loader = torch.utils.data.DataLoader(face_valdataset, **dataloader_kwargs)
    
        body_valdataset = Comic2kDataset(
            data_dir=self.body_data_dir,
            train=False,
            img_size=self.test_size,
            preproc=ValTransform(legacy=False),
        )
        
        body_sampler = torch.utils.data.SequentialSampler(body_valdataset)
        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": False,
            "sampler": body_sampler,
        }
        
        dataloader_kwargs["batch_size"] = batch_size
        body_val_loader = torch.utils.data.DataLoader(body_valdataset, **dataloader_kwargs)

        return face_val_loader, body_val_loader

    def get_evaluator(self, batch_size, is_distributed):
        
        from yolox.evaluators import ComicEvaluator

        face_val_loader, body_val_loader = self.get_eval_loader(batch_size, is_distributed)
        
        face_evaluator = ComicEvaluator(
            dataloader=face_val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=1,
        )
        
        body_evaluator = ComicEvaluator(
            dataloader=body_val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=1,
        )
        
        return face_evaluator, body_evaluator

    def eval(self, model, evaluator, is_distributed=False, mode=1, **kwargs):
        return evaluator.evaluate(model, is_distributed, False, mode=mode, **kwargs)
    
    def __repr__(self):
        table_header = ["keys", "values"]
        exp_table = [
            (str(k), pprint.pformat(v))
            for k, v in vars(self).items()
            if not k.startswith("_")
        ]
        return tabulate(exp_table, headers=table_header, tablefmt="fancy_grid")

    def merge(self, cfg_list):
        assert len(cfg_list) % 2 == 0
        for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
            # only update value with same key
            if hasattr(self, k):
                src_value = getattr(self, k)
                src_type = type(src_value)
                if src_value is not None and src_type != type(v):
                    try:
                        v = src_type(v)
                    except Exception:
                        v = ast.literal_eval(v)
                setattr(self, k, v)
    
    
    def random_resize(self, data_loader, epoch, rank, is_distributed=False):
        
        tensor = torch.LongTensor(2).cuda()

        if rank == 0:
            size_factor = self.input_size[1] * 1.0 / self.input_size[0]
            
            if not hasattr(self, 'random_size'):
                min_size = int(self.input_size[0] / 32) - self.multiscale_range
                max_size = int(self.input_size[0] / 32) + self.multiscale_range
                self.random_size = (min_size, max_size)

            size = random.randint(*self.random_size)
            size = (int(32 * size), 32 * int(size * size_factor))
            tensor[0] = size[0]
            tensor[1] = size[1]

        input_size = (tensor[0].item(), tensor[1].item())
        return input_size

    
    def preprocess(self, inputs, targets, tsize):
        scale_y = tsize[0] / self.input_size[0]
        scale_x = tsize[1] / self.input_size[1]
        if scale_x != 1 or scale_y != 1:
            inputs = nn.functional.interpolate(
                inputs, size=tsize, mode="bilinear", align_corners=False
            )
            targets[..., 1::2] = targets[..., 1::2] * scale_x
            targets[..., 2::2] = targets[..., 2::2] * scale_y
        return inputs, targets

#     @abstractmethod
#     def get_lr_scheduler(
#         self, lr: float, iters_per_epoch: int, **kwargs
#     ) -> LRScheduler:
#         pass
    
#     def get_lr_scheduler(self, lr, iters_per_epoch):
#         from yolox.utils import LRScheduler

#         scheduler = LRScheduler(
#             self.scheduler,
#             lr,
#             iters_per_epoch,
#             self.max_epoch,
#             warmup_epochs=self.warmup_epochs,
#             warmup_lr_start=self.warmup_lr,
#             no_aug_epochs=self.no_aug_epochs,
#             min_lr_ratio=self.min_lr_ratio,
#         )
#         return scheduler