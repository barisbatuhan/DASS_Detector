#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import os
import random

import torch
import torch.distributed as dist
import torch.nn as nn

from .base_exp import BaseExp


class UnsupervisedExp(BaseExp):
    def __init__(self, model_size, head_mode, **kwargs):
        super().__init__(model_size)
        
        # ---------------- model config ---------------- #
        
        self.head_mode = head_mode

        # ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        self.data_num_workers = 4
        self.weak_input_size  = (1024, 1024)
        self.input_size       = (640, 640)  # (height, width)
        # Actual multiscale ranges: [640-5*32, 640+5*32].
        # To disable multiscale training, set the
        # self.multiscale_range to 0.
        self.multiscale_range = 5

        self.train_data_dir = {
            "icf": self.face_data_dir["icf_train_imgs"],
            "m109": os.path.join(self.face_data_dir["m109"], "images"),
            "comic": os.path.join(self.body_data_dir["comic"], "JPEGImages/"), 
            "watercolor": os.path.join(self.body_data_dir["watercolor"], "JPEGImages/"), 
            "clipart": os.path.join(self.body_data_dir["clipart"], "JPEGImages/"),
            "golden": self.face_data_dir["golden_pages"]
        }
        
        # --------------- transform config ----------------- #
        self.hsv_prob            = 1.0
        self.flip_prob           = 0.5
        self.vertical_flip_prob  = 0.12
        
        self.gaussian_noise_prob = 0.3
        self.crop_prob           = 1.0
        
        ##########################################
        ## CURRENTLY UNAVAILABLE OPTIONS:
        ##########################################
        # self.mosaic_prob         = 0.0
        # self.perp_rotate_prob    = 0.15
        # self.degrees             = 30.0
        # self.translate           = 0.1
        # self.mosaic_scale        = (0.1, 2)
        # self.shear               = 2.0

        # ----------------- teacher config ------------------- #
        self.teacher_face_conf_thold = 0.65
        self.teacher_face_nms_thold  = 0.2
        
        self.teacher_body_conf_thold = 0.65
        self.teacher_body_nms_thold  = 0.4
        
        self.face_num_iter           = 1
        self.body_num_iter           = 1
        
        self.match_models_per_iter   = 500 # set to 0 for no equalization
        self.const_ema_rate          = True
        self.ema_keep_rate           = 0.9996
        self.update_teacher_per_iter = 1
        
        self.ema_exp_denominator     = 2000
        self.reverse                 = False
        
        self.use_focal_loss          = False
        
        # --------------- student OHEM config ----------------- #
        self.num_neg_ratio     = 3
        self.select_neg_random = False
        
        self.upper_conf_thold_start  = 0.5
        self.upper_conf_thold_end    = 0.5
        self.upper_conf_thold_step   = 0.0
        
        self.lower_conf_thold_start  = 0.5
        self.lower_conf_thold_end    = 0.5
        self.lower_conf_thold_step   = 0.0
        
        self.upper_conf_thold = self.upper_conf_thold_start
        self.lower_conf_thold = self.lower_conf_thold_start
        
        
        self.upper_iou_thold  = 0.5
        self.lower_iou_thold  = 0.35
        
        self.reg_loss_coeff   = 2
        
        # --------------  training config --------------------- #
        
        self.lr           = 0.0001
        self.warmup_lr    = self.lr

        self.warmup_epochs = 0
        self.max_epoch     = 80
        self.no_aug_epochs = 0
        self.max_iter      = 100
        self.print_interval = 10
        
        self.l1_loss_start = self.max_epoch - self.no_aug_epochs

    def change_conf_tholds(self):
        
        if not self.use_focal_loss:
            
            if self.upper_conf_thold > self.upper_conf_thold_end:
                self.upper_conf_thold -= self.upper_conf_thold_step
                self.model.face_head.loss_fn.upper_conf_thold = self.upper_conf_thold
                self.model.body_head.loss_fn.upper_conf_thold = self.upper_conf_thold
            
            if self.lower_conf_thold < self.lower_conf_thold_end:
                self.lower_conf_thold += self.lower_conf_thold_step
                self.model.face_head.loss_fn.lower_conf_thold = self.lower_conf_thold
                self.model.body_head.loss_fn.lower_conf_thold = self.lower_conf_thold
            
    
    def get_loss_fn(self):
        from yolox.models import get_loss_fn
        
        if not self.use_focal_loss:
            loss_fn = get_loss_fn("unsupervised")
            return loss_fn(
                num_neg_ratio=self.num_neg_ratio,
                upper_conf_thold=self.upper_conf_thold, 
                lower_conf_thold=self.lower_conf_thold,
                upper_iou_thold=self.upper_iou_thold,
                lower_iou_thold=self.lower_iou_thold,
                reg_loss_coeff=self.reg_loss_coeff,
                random_select=self.select_neg_random
            )
        else:
            loss_fn = get_loss_fn("yolox")
            return loss_fn(strides=self.strides, in_channels=self.in_channels)
    
    
    def get_optimizer(self, batch_size: int) -> torch.optim.Optimizer:
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.lr

            pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)  # biases
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    pg0.append(v.weight)  # no decay
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg1.append(v.weight)  # apply decay

            optimizer = torch.optim.SGD(pg0, lr=lr) # regular SGD is used
            
            optimizer.add_param_group(
                {"params": pg1, "weight_decay": self.weight_decay}
            )  # add pg1 with weight_decay
            optimizer.add_param_group({"params": pg2})
            self.optimizer = optimizer

        return self.optimizer
    
    
    def get_data_loader(
        self, batch_size, is_distributed=False, no_aug=False, cache_img=False
    ):
        from yolox.data import (
            RawUnsupervisedComicsDataset,
            StrongTransform,
            WeakTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            worker_init_reset_seed,
        )
        from yolox.utils import (
            wait_for_the_master,
            get_local_rank,
        )

        dataset = RawUnsupervisedComicsDataset(
            self.train_data_dir,
            train=True,
            weak_img_size=self.weak_input_size,
            strong_img_size=self.input_size,
            weak_preproc=WeakTransform(flip_prob=self.flip_prob),
            strong_preproc=StrongTransform(
                flip_prob=self.flip_prob, 
                vertical_flip_prob=self.vertical_flip_prob,
                hsv_prob=self.hsv_prob,
                crop_prob=self.crop_prob,
                gaussian_noise_prob=self.gaussian_noise_prob),
        )

        self.dataset = dataset

        sampler = InfiniteSampler(len(self.dataset), 
                                  seed=self.seed if self.seed else 0)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": False}
        dataloader_kwargs["batch_sampler"] = batch_sampler
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed
        train_loader = DataLoader(self.dataset, **dataloader_kwargs)
        
        return train_loader

    
##############################################################################

"""
EXPERIMENT:
---------------
CHANGING MATCHING OF TEACHER AND STUDENT NETWORK IN DIFFERENT ITERATIONS 
WHILE KEEPING CONFIDENCE LEVELS OF THE STUDENT NETWORK THE SAME.
""" 

class Match500ConstUnsupervisedExp(UnsupervisedExp):
    def __init__(self, model_size, head_mode, **kwargs):
        super().__init__(model_size, head_mode)
        self.match_models_per_iter   = 500
        self.upper_conf_thold_start  = 0.15
        self.upper_conf_thold_end    = self.upper_conf_thold_start
        self.upper_conf_thold_step   = 0.0
        self.lower_conf_thold_start  = 0.85
        self.lower_conf_thold_end    = self.lower_conf_thold_start
        self.lower_conf_thold_step   = 0.0
    
class Match250ConstUnsupervisedExp(Match500ConstUnsupervisedExp):
    def __init__(self, model_size, head_mode, **kwargs):
        super().__init__(model_size, head_mode)
        self.match_models_per_iter   = 250

class Match1000ConstUnsupervisedExp(Match500ConstUnsupervisedExp):
    def __init__(self, model_size, head_mode, **kwargs):
        super().__init__(model_size, head_mode)
        self.match_models_per_iter   = 1000

class Match2000ConstUnsupervisedExp(Match500ConstUnsupervisedExp):
    def __init__(self, model_size, head_mode, **kwargs):
        super().__init__(model_size, head_mode)
        self.match_models_per_iter   = 2000
        
class Match5000ConstUnsupervisedExp(Match500ConstUnsupervisedExp):
    def __init__(self, model_size, head_mode, **kwargs):
        super().__init__(model_size, head_mode)
        self.match_models_per_iter   = 5000

class NoMatchConstUnsupervisedExp(Match500ConstUnsupervisedExp):
    def __init__(self, model_size, head_mode, **kwargs):
        super().__init__(model_size, head_mode)
        self.match_models_per_iter   = 0
        # self.lr           = 0.000001

##############################################################################

"""
EXPERIMENT:
---------------
USING MODIFIED FOCAL LOSS OF YOLOX NETWORK INSTEAD OF OHEM LOSS
WHILE KEEPING CONFIDENCE LEVELS OF THE STUDENT NETWORK THE SAME.
"""     
        
class FocalConstUnsupervisedExp(Match500ConstUnsupervisedExp): # tends to change
    def __init__(self, model_size, head_mode, **kwargs):
        super().__init__(model_size, head_mode)
        self.use_focal_loss          = True

class FocalNoMatchConstUnsupervisedExp(Match500ConstUnsupervisedExp): # tends to change
    def __init__(self, model_size, head_mode, **kwargs):
        super().__init__(model_size, head_mode)
        self.use_focal_loss          = True
        
##############################################################################

"""
EXPERIMENT:
---------------
KEEPING CONFIDENCE LEVELS OF THE STUDENT NETWORK DYNAMIC BY LINEARLY 
CHANGING ITS VALUES WITH THE STEP SIZE OF 0.0002.
""" 

class MovingUnsupervisedExp(Match500ConstUnsupervisedExp): # tends to change
    def __init__(self, model_size, head_mode, **kwargs):
        super().__init__(model_size, head_mode)
        self.upper_conf_thold_start  = 0.5
        self.upper_conf_thold_end    = 0.1
        self.upper_conf_thold_step   = 0.0002
        self.lower_conf_thold_start  = 0.5
        self.lower_conf_thold_end    = 0.9
        self.lower_conf_thold_step   = 0.0002
        
##############################################################################

"""
EXPERIMENT:
---------------
TESTING THE MODEL WITH DIFFERENT EMA KEEP RATES TO SEE WHICH VALUE IS MORE
SUITABLE TO THE ARCHITECTURE.
""" 

class EMA9990UnsupervisedExp(Match500ConstUnsupervisedExp): # tends to change
    def __init__(self, model_size, head_mode, **kwargs):
        super().__init__(model_size, head_mode)
        self.ema_keep_rate           = 0.999
        
class EMA9992UnsupervisedExp(Match500ConstUnsupervisedExp): # tends to change
    def __init__(self, model_size, head_mode, **kwargs):
        super().__init__(model_size, head_mode)
        self.ema_keep_rate           = 0.9992

class EMA9996UnsupervisedExp(Match500ConstUnsupervisedExp): # tends to change
    def __init__(self, model_size, head_mode, **kwargs):
        super().__init__(model_size, head_mode)
        self.ema_keep_rate           = 0.9996

class EMA9998UnsupervisedExp(Match500ConstUnsupervisedExp): # tends to change
    def __init__(self, model_size, head_mode, **kwargs):
        super().__init__(model_size, head_mode)
        self.ema_keep_rate           = 0.9998

class EMA9999UnsupervisedExp(Match500ConstUnsupervisedExp): # tends to change
    def __init__(self, model_size, head_mode, **kwargs):
        super().__init__(model_size, head_mode)
        self.ema_keep_rate           = 0.9999
        
##############################################################################

"""
EXPERIMENT:
---------------
TESTING THE MODEL WITH DIFFERENT REGRESSION LOSS WEIGHTS
""" 

class NoRegUnsupervisedExp(Match500ConstUnsupervisedExp): # tends to change
    def __init__(self, model_size, head_mode, **kwargs):
        super().__init__(model_size, head_mode)
        self.reg_loss_coeff   = 0

class Reg1UnsupervisedExp(Match500ConstUnsupervisedExp): # tends to change
    def __init__(self, model_size, head_mode, **kwargs):
        super().__init__(model_size, head_mode)
        self.reg_loss_coeff   = 1
        
class Reg4UnsupervisedExp(Match500ConstUnsupervisedExp): # tends to change
    def __init__(self, model_size, head_mode, **kwargs):
        super().__init__(model_size, head_mode)
        self.reg_loss_coeff   = 4
        
class Reg10UnsupervisedExp(Match500ConstUnsupervisedExp): # tends to change
    def __init__(self, model_size, head_mode, **kwargs):
        super().__init__(model_size, head_mode)
        self.reg_loss_coeff   = 10
        
##############################################################################

"""
EXPERIMENT:
---------------
TESTING THE MODEL WITH DIFFERENT REGRESSION LOSS WEIGHTS
""" 

class StuPos50Neg50UnsupervisedExp(Match500ConstUnsupervisedExp): # tends to change
    def __init__(self, model_size, head_mode, **kwargs):
        super().__init__(model_size, head_mode)
        self.upper_conf_thold_start  = 0.50
        self.upper_conf_thold_end    = self.upper_conf_thold_start
        self.upper_conf_thold_step   = 0.0
        self.lower_conf_thold_start  = 0.50
        self.lower_conf_thold_end    = self.lower_conf_thold_start
        self.lower_conf_thold_step   = 0.0
    
class StuPos70Neg30UnsupervisedExp(Match500ConstUnsupervisedExp): # tends to change
    def __init__(self, model_size, head_mode, **kwargs):
        super().__init__(model_size, head_mode)
        self.upper_conf_thold_start  = 0.70
        self.upper_conf_thold_end    = self.upper_conf_thold_start
        self.upper_conf_thold_step   = 0.0
        self.lower_conf_thold_start  = 0.30
        self.lower_conf_thold_end    = self.lower_conf_thold_start
        self.lower_conf_thold_step   = 0.0

class StuPos30Neg70UnsupervisedExp(Match500ConstUnsupervisedExp): # tends to change
    def __init__(self, model_size, head_mode, **kwargs):
        super().__init__(model_size, head_mode)
        self.upper_conf_thold_start  = 0.30
        self.upper_conf_thold_end    = self.upper_conf_thold_start
        self.upper_conf_thold_step   = 0.0
        self.lower_conf_thold_start  = 0.70
        self.lower_conf_thold_end    = self.lower_conf_thold_start
        self.lower_conf_thold_step   = 0.0
        
class StuPos05Neg95UnsupervisedExp(Match500ConstUnsupervisedExp): # tends to change
    def __init__(self, model_size, head_mode, **kwargs):
        super().__init__(model_size, head_mode)
        self.upper_conf_thold_start  = 0.05
        self.upper_conf_thold_end    = self.upper_conf_thold_start
        self.upper_conf_thold_step   = 0.0
        self.lower_conf_thold_start  = 0.95
        self.lower_conf_thold_end    = self.lower_conf_thold_start
        self.lower_conf_thold_step   = 0.0

class StuPos00Neg100UnsupervisedExp(Match500ConstUnsupervisedExp): # tends to change
    def __init__(self, model_size, head_mode, **kwargs):
        super().__init__(model_size, head_mode)
        self.upper_conf_thold_start  = 0.00
        self.upper_conf_thold_end    = self.upper_conf_thold_start
        self.upper_conf_thold_step   = 0.0
        self.lower_conf_thold_start  = 1.00
        self.lower_conf_thold_end    = self.lower_conf_thold_start
        self.lower_conf_thold_step   = 0.0
        
class TeacConf50UnsupervisedExp(Match500ConstUnsupervisedExp): # tends to change
    def __init__(self, model_size, head_mode, **kwargs):
        super().__init__(model_size, head_mode)
        self.teacher_face_conf_thold = 0.5
        self.teacher_body_conf_thold = 0.5
        
class TeacConf75UnsupervisedExp(Match500ConstUnsupervisedExp): # tends to change
    def __init__(self, model_size, head_mode, **kwargs):
        super().__init__(model_size, head_mode)
        self.teacher_face_conf_thold = 0.75
        self.teacher_body_conf_thold = 0.75
        
class TeacConf35UnsupervisedExp(Match500ConstUnsupervisedExp): # tends to change
    def __init__(self, model_size, head_mode, **kwargs):
        super().__init__(model_size, head_mode)
        self.teacher_face_conf_thold = 0.35
        self.teacher_body_conf_thold = 0.35

class TeacConf90UnsupervisedExp(Match500ConstUnsupervisedExp): # tends to change
    def __init__(self, model_size, head_mode, **kwargs):
        super().__init__(model_size, head_mode)
        self.teacher_face_conf_thold = 0.9
        self.teacher_body_conf_thold = 0.9