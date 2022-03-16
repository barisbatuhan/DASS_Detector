#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import os
import random

import torch
import torch.distributed as dist
import torch.nn as nn

from .base_exp import BaseExp


class SingleDatasetExp(BaseExp):
    def __init__(self, model_size, head_mode, dataset_name=None):
        super().__init__(model_size)
        
        # ---------------- model config ---------------- #
        
        self.dataset_name = dataset_name
        self.head_mode    = head_mode

        # ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        self.data_num_workers = 4
        self.input_size = (640, 640)  # (height, width)
        # Actual multiscale ranges: [640-5*32, 640+5*32].
        # To disable multiscale training, set the
        # self.multiscale_range to 0.
        self.multiscale_range = 5
        
        self.image_limit  = None
        self.train_mode   = None
        
        self.training_comic_model = True

        # --------------- transform config ----------------- #
        
        self.degrees      = 30.0           # rotation to the image randomly between [-degree, degree]
        self.translate    = 0.1            # ratio of how much to move the mosaic center from the actual center
        self.mosaic_scale = (0.99, 1.5)    # how to scale each image in a mosaic, if set to 0.5, then the mosaic image is completely visible
        self.mixup_scale  = (0.5, 1.5)     # scale of the image chosen for mixup. 
        self.shear        = 2.0            # bends the image to a certain direction, processes the plane in a 3D space
        self.mosaic_prob  = 0.75           # probability of having mosaic augmentation
        self.mixup_prob   = 0.5            # probability of having mixup augmentation
        self.hsv_prob     = 1.0            # probability of color distortion of the entire image
        self.flip_prob    = 0.5            # probability of horizontal rotation of entire image
        self.vertical_flip_prob = 0.12     # probability of vertical rotation of entire image
        self.perp_rotate_prob   = 0.15     # probability of 90 or 270 degree rotation of the entire image
        self.rand_mosaic_center = False    # if set to False, mosaic center is center of image, if True, then it is between [0.5*center, 1.5*center]
        
        if self.training_comic_model:
            self.enable_mixup = True           # whether to use mixup or not
            self.speech_bubble_prob = 0.0      # probability of drawing speech bubble to the image, only works in mosaic mode
        
        else:
            self.enable_mixup = False          # whether to use mixup or not
            self.speech_bubble_prob = 0.5      # probability of drawing speech bubble to the image, only works in mosaic mode
            # opts: 
            # - "all" for all objs, 
            # - "person" for persons, 
            # - "living" for all persons and animals, 
            # - "nobird" for all persons and animals except birds
            self.coco_obj_choice = "living" 

        # --------------  training config --------------------- #
        self.warmup_epochs = 5
        self.max_epoch     = 300
        self.no_aug_epochs = 15
        
        self.l1_loss_start = self.max_epoch - self.no_aug_epochs


    def get_loss_fn(self):
        from yolox.models import get_loss_fn
        loss_fn = get_loss_fn("yolox")
        return loss_fn(strides=self.strides, in_channels=self.in_channels)
    
    
    def get_data_loader(
        self, batch_size, is_distributed=False, no_aug=False, cache_img=False
    ):
        
        from yolox.data import (
            Manga109Dataset,
            ICartoonFaceDataset,
            DCM772FramesDataset,
            Manga109FramesDataset,
            COCOPersonsDataset,
            WiderFaceDataset,
            Comic2kDataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            worker_init_reset_seed,
        )
        
        dataset_obj, data_dir = None, None
        
        if self.dataset_name == "icartoonface":
            dataset_obj = ICartoonFaceDataset
            data_dir    = self.face_data_dir
            self.train_mode = "face"
        elif self.dataset_name == "manga109":
            dataset_obj = Manga109FramesDataset
            data_dir    = self.face_data_dir
            self.train_mode = "both"
        elif self.dataset_name == "dcm772":
            dataset_obj = DCM772FramesDataset
            data_dir    = self.face_data_dir
            self.train_mode = "both"
        elif self.dataset_name == "comic2k":
            dataset_obj = Comic2kDataset
            data_dir    = self.body_data_dir
            self.train_mode = "body"
        elif self.head_mode == 2:
            dataset_obj = COCOPersonsDataset
            data_dir    = self.body_data_dir["coco"]
            self.train_mode = "body"
        elif self.head_mode == 1:
            dataset_obj = WiderFaceDataset
            data_dir    = self.face_data_dir
            self.train_mode = "face"
        
        dataset = dataset_obj(
            data_dir=data_dir,
            train=True,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob),
            limit_dataset=self.image_limit
        )
        
        if hasattr(dataset, "random_style"):
            dataset.set_style(self.dataset_name)
            dataset.set_allowed_obj_ids(self.coco_obj_choice)
        
        dataset = MosaicDetection(
            dataset,
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
        

        self.dataset = dataset

        sampler = InfiniteSampler(len(self.dataset), 
                                  seed=self.seed if self.seed else 0)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )
        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed
        
        train_loader = DataLoader(self.dataset, **dataloader_kwargs)
        
        return train_loader
    
    
    def get_eval_loader(self, batch_size, is_distributed):

        from yolox.data import ICartoonFaceDataset, Manga109Dataset, DCM772Dataset, Comic2kDataset, ValTransform
        
        if self.dataset_name == "icartoonface":
            face_dataset_obj = ICartoonFaceDataset
            body_dataset_obj = None
            data_dir         = self.face_data_dir
            self.eval_face_select_idx = 0
            self.eval_body_select_idx = None
        
        elif self.dataset_name == "manga109":
            face_dataset_obj = Manga109Dataset
            body_dataset_obj = Manga109Dataset
            data_dir         = self.face_data_dir
            self.eval_face_select_idx = 1
            self.eval_body_select_idx = 2
            
        
        elif self.dataset_name == "dcm772":
            face_dataset_obj = DCM772Dataset
            body_dataset_obj = DCM772Dataset
            data_dir         = self.face_data_dir
            self.eval_face_select_idx = 1
            self.eval_body_select_idx = 2
        
        elif self.dataset_name == "comic2k":
            face_dataset_obj = None
            body_dataset_obj = Comic2kDataset
            data_dir         = self.body_data_dir
            self.eval_face_select_idx = None
            self.eval_body_select_idx = 0
        
        elif self.head_mode == 2:
            face_dataset_obj = None
            body_dataset_obj = Comic2kDataset
            data_dir         = self.body_data_dir
            self.eval_face_select_idx = None
            self.eval_body_select_idx = 0
        
        elif self.head_mode == 1:
            face_dataset_obj = ICartoonFaceDataset
            body_dataset_obj = None
            data_dir         = self.face_data_dir
            self.eval_face_select_idx = 0
            self.eval_body_select_idx = None

        if face_dataset_obj is not None:
            face_valdataset = face_dataset_obj(
                data_dir=data_dir,
                train=False,
                img_size=self.test_size,
                preproc=ValTransform(legacy=False),
            )

            face_sampler = torch.utils.data.SequentialSampler(face_valdataset)
            dataloader_kwargs = {
                "num_workers": self.data_num_workers,
                "pin_memory": True,
                "sampler": face_sampler,
            }
            
            dataloader_kwargs["batch_size"] = batch_size
            face_val_loader = torch.utils.data.DataLoader(face_valdataset, **dataloader_kwargs)
        
        else:
            face_val_loader = None

        if body_dataset_obj is not None:
            body_valdataset = body_dataset_obj(
                data_dir=data_dir,
                train=False,
                img_size=self.test_size,
                preproc=ValTransform(legacy=False),
            )
            
            body_sampler = torch.utils.data.SequentialSampler(body_valdataset)
            dataloader_kwargs = {
                "num_workers": self.data_num_workers,
                "pin_memory": True,
                "sampler": body_sampler,
            }
            
            dataloader_kwargs["batch_size"] = batch_size
            body_val_loader = torch.utils.data.DataLoader(body_valdataset, **dataloader_kwargs)
        
        else:
            body_val_loader = None
    
        return face_val_loader, body_val_loader
    
    
    def get_evaluator(self, batch_size, is_distributed):
        
        from yolox.evaluators import ComicEvaluator

        face_val_loader, body_val_loader = self.get_eval_loader(batch_size, is_distributed)
        
        if face_val_loader is not None:
            
            face_evaluator = ComicEvaluator(
                dataloader=face_val_loader,
                img_size=self.test_size,
                confthre=self.test_conf,
                nmsthre=self.nmsthre,
                num_classes=1,
            )
        
        else:
            face_evaluator = None
        
        if body_val_loader is not None:
            
            body_evaluator = ComicEvaluator(
                dataloader=body_val_loader,
                img_size=self.test_size,
                confthre=self.test_conf,
                nmsthre=self.nmsthre,
                num_classes=1,
            )
        
        else:
            body_evaluator = None
        
        return face_evaluator, body_evaluator
    
    def eval(self, model, evaluator, is_distributed=False, mode=1, **kwargs):
        
        if evaluator is None:
            return None
        else:
            if mode == 1:
                select_index = self.eval_face_select_idx
            elif mode == 2:
                select_index = self.eval_body_select_idx
            else:
                raise ValueError
                
            return evaluator.evaluate(model, is_distributed, False, 
                                      mode=mode, 
                                      select_index=select_index,
                                      **kwargs)