#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import os
import random

import torch
import torch.distributed as dist
import torch.nn as nn

from .base_exp import BaseExp

class StyledExp(BaseExp):
    def __init__(self, model_size, head_mode, dataset_name=None):
        super().__init__(model_size)
        
        # ---------------- model config ---------------- #
        
        self.head_mode = head_mode

        # ---------------- dataloader config ---------------- #
        self.data_num_workers = 4
        
        self.input_size = (640, 640)  # (height, width)
        # Actual multiscale ranges: [640-5*32, 640+5*32].
        # To disable multiscale training, set the self.multiscale_range to 0.
        self.multiscale_range = 5

        # --------------- transform config ----------------- #
        
        self.degrees      = 20.0           # rotation to the image randomly between [-degree, degree]
        self.translate    = 0.1            # ratio of how much to move the mosaic center from the actual center
        self.mosaic_scale = (0.99, 1.5)    # how to scale each image in a mosaic, if set to 0.5, then the mosaic image is completely visible
        self.mixup_scale  = (0.5, 1.5)     # scale of the image chosen for mixup. 
        self.shear        = 2.0            # bends the image to a certain direction, processes the plane in a 3D space
        self.mosaic_prob  = 0.75           # probability of having mosaic augmentation
        self.mixup_prob   = 0.5            # probability of having mixup augmentation
        self.enable_mixup = False          # whether to use mixup or not
        self.hsv_prob     = 1.0            # probability of color distortion of the entire image
        self.flip_prob    = 0.5            # probability of horizontal rotation of entire image
        self.vertical_flip_prob = 0.05     # probability of vertical rotation of entire image
        self.perp_rotate_prob   = 0.00     # probability of 90 or 270 degree rotation of the entire image
        self.rand_mosaic_center = False    # if set to False, mosaic center is center of image, if True, then it is between [0.5*center, 1.5*center]
        self.speech_bubble_prob = 0.5      # probability of drawing speech bubble to the image, only works in mosaic mode
        
        # --------------  training config --------------------- #
        self.warmup_epochs = 0
        self.max_epoch     = 350
        self.no_aug_epochs = 15
        
        self.l1_loss_start = self.max_epoch - self.no_aug_epochs
        
        # opts: 
        # - "all" for all objs, 
        # - "person" for persons, 
        # - "living" for all persons and animals, 
        # - "nobird" for all persons and animals except birds
        self.coco_obj_choice = "living"  
        
        self.face_random_style = None # will be set by inherited objects
        self.body_random_style = None # will be set by inherited objects
        

    def get_loss_fn(self):
        from yolox.models import get_loss_fn
        loss_fn = get_loss_fn("yolox")
        return loss_fn(strides=self.strides, in_channels=self.in_channels)
    
    def get_data_loader(
        self, batch_size, is_distributed=False, no_aug=False, cache_img=False
    ):
        from yolox.data import (
            WiderFaceDataset,
            COCOPersonsDataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            worker_init_reset_seed,
        )

        face_dataset = WiderFaceDataset(
            data_dir=self.face_data_dir,
            train=True,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                vertical_flip_prob=self.vertical_flip_prob,
                hsv_prob=self.hsv_prob),
            filter_file="datasets/widerface_filtered_1-60.txt",
            random_style=self.face_random_style,
        )

        face_dataset = MosaicDetection(
            face_dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                vertical_flip_prob=self.vertical_flip_prob,
                hsv_prob=self.hsv_prob),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
            rand_mosaic_center=self.rand_mosaic_center,
            speech_bubble_prob=self.speech_bubble_prob,
            perp_rotate_prob=self.perp_rotate_prob,
        )

        self.face_dataset = face_dataset

        face_sampler = InfiniteSampler(len(self.face_dataset), 
                                       seed=self.seed if self.seed else 0)

        face_batch_sampler = YoloBatchSampler(
            sampler=face_sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers,
                             "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = face_batch_sampler
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed
        face_train_loader = DataLoader(self.face_dataset, **dataloader_kwargs)
        
        
        body_dataset = COCOPersonsDataset(
            data_dir=self.body_data_dir["coco"],
            json_file="instances_train2017.json",
            name="train2017",
            img_size=self.input_size,
            random_style=self.body_random_style,
            filter_file="datasets/yolox_coco_train_person_ids.txt",
            filter_obj_style=self.coco_obj_choice,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                vertical_flip_prob=self.vertical_flip_prob,
                hsv_prob=self.hsv_prob),
            cache=False,
        )

        body_dataset = MosaicDetection(
            body_dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                vertical_flip_prob=self.vertical_flip_prob,
                hsv_prob=self.hsv_prob),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
            rand_mosaic_center=self.rand_mosaic_center,
            speech_bubble_prob=self.speech_bubble_prob,
            perp_rotate_prob=self.perp_rotate_prob,
        )

        self.body_dataset = body_dataset

        body_sampler = InfiniteSampler(len(self.body_dataset), 
                                       seed=self.seed if self.seed else 0)

        body_batch_sampler = YoloBatchSampler(
            sampler=body_sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, 
                             "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = body_batch_sampler
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed
        body_train_loader = DataLoader(self.body_dataset, **dataloader_kwargs)

        return face_train_loader, body_train_loader

    
"""
Available styles:
cartoongan_hayao    cartoongan_shinkai  cyclegan_vangogh  ganilla_miyazaki
cartoongan_hosoda   cyclegan_cezanne    ganilla_AS        original
cartoongan_paprika  cyclegan_monet      ganilla_KH        whitebox
"""

class AllStyledExp(StyledExp):
    def __init__(self, model_size, head_mode, dataset_name=None):
        super().__init__(model_size, head_mode)
        self.face_random_style = "all"
        self.body_random_style = "all"
        
class AllOnlyPersonStyledExp(StyledExp):
    def __init__(self, model_size, head_mode, dataset_name=None):
        super().__init__(model_size, head_mode)
        self.face_random_style = "all"
        self.body_random_style = "all"
        self.coco_obj_choice   = "person"

class AllNoBubbleStyledExp(StyledExp):
    def __init__(self, model_size, head_mode, dataset_name=None):
        super().__init__(model_size, head_mode)
        self.face_random_style  = "all"
        self.body_random_style  = "all"
        self.speech_bubble_prob = 0.00

class NoneStyledExp(StyledExp):
    def __init__(self, model_size, head_mode, dataset_name=None):
        super().__init__(model_size, head_mode)
        self.face_random_style = "original"
        self.body_random_style = "original"
        
class BestStyledExp(StyledExp):
    def __init__(self, model_size, head_mode, dataset_name=None):
        super().__init__(model_size, head_mode)
        self.face_random_style = "best_comb"
        self.body_random_style = "best_comb"

class CustomStyledExp(StyledExp):
    def __init__(self, model_size, head_mode, dataset_name=None):
        super().__init__(model_size, head_mode)
        assert dataset_name is not None
        self.face_random_style = dataset_name
        self.body_random_style = dataset_name