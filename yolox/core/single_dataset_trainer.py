#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import datetime
import os
import time
from loguru import logger

import numpy as np

import torch

from yolox.data import DataPrefetcher
from yolox.utils import (
    ModelEMA,
    get_model_info,
    load_ckpt,
    occupy_mem,
    save_checkpoint,
    setup_logger,
    all_reduce_norm,
)

from .base_trainer import BaseTrainer

class SingleDatasetTrainer(BaseTrainer):
    
    def __init__(self, exp, args):
        super().__init__(exp, args)
        
    def train_one_iter(self):
        
        iter_start_time = time.time()
        
        inps, targets = self.prefetcher.next()
        inps = inps.to(self.data_type)
        targets = targets.to(self.data_type)
        targets.requires_grad = False
        inps, targets = self.exp.preprocess(inps, targets, self.input_size)
        data_end_time = time.time()
        
        outputs = {}
           
        # face update
        if self.exp.head_mode in [0, 1] and self.exp.train_mode in ["face", "both"]:
            
            if self.exp.train_mode == "both":
                ftargets = torch.zeros_like(targets)
                face_idx = self.train_loader.dataset._dataset.class_dict["face"]
                
                nonzero_idx = []
                for i in range(targets.shape[0]):
                    inst_indices = torch.where(targets[i,:,0] == face_idx)[0]
                    if inst_indices is not None and len(inst_indices) > 0:
                        ftargets[i,:len(inst_indices), ...] = targets[i, inst_indices, ...]
                        nonzero_idx.append(i)
     
                nonzero_idx = torch.LongTensor(nonzero_idx)
                ftargets    = ftargets[nonzero_idx, ...]
                fimgs       = inps[nonzero_idx, ...]

            else:
                fimgs, ftargets = inps, targets

            if fimgs is not None and ftargets is not None and ftargets.shape[0] > 0:

                with torch.cuda.amp.autocast(enabled=self.amp_training):
                    foutputs = self.model(fimgs, targets=ftargets, mode=1)

                loss = foutputs["total_loss"]
    
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
    
                if self.use_model_ema:
                    self.ema_model.update(self.model)
                    
                outputs.update({
                    "floss": foutputs["total_loss"],
                    "fiou_loss": foutputs["iou_loss"],
                    "fl1_loss": foutputs["l1_loss"],
                    "fconf_loss": foutputs["conf_loss"]
                })
                
        self.model.zero_grad()

        # body_update
        if self.exp.head_mode in [0, 2] and self.exp.train_mode in ["body", "both"]:
            
            if self.exp.train_mode == "both":
                btargets = torch.zeros_like(targets)
                body_idx = self.train_loader.dataset._dataset.class_dict["body"]
                
                nonzero_idx = []
                for i in range(targets.shape[0]):
                    inst_indices = torch.where(targets[i,:,0] == body_idx)[0]
                    if inst_indices is not None and len(inst_indices) > 0:
                        btargets[i,:len(inst_indices), ...] = targets[i, inst_indices, ...]
                        nonzero_idx.append(i)
                        
                nonzero_idx = torch.LongTensor(nonzero_idx)
                btargets    = btargets[nonzero_idx, ...]
                bimgs       = inps[nonzero_idx, ...]

            else:
                bimgs, btargets = inps, targets

            if bimgs is not None and btargets is not None and btargets.shape[0] > 0:

                with torch.cuda.amp.autocast(enabled=self.amp_training):
                    boutputs = self.model(bimgs, targets=btargets, mode=2)

                loss = boutputs["total_loss"]
    
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
        
                if self.use_model_ema:
                    self.ema_model.update(self.model)
    
                outputs.update({
                    "bloss": boutputs["total_loss"],
                    "biou_loss": boutputs["iou_loss"],
                    "bl1_loss": boutputs["l1_loss"],
                    "bconf_loss": boutputs["conf_loss"]
                })

        iter_end_time = time.time()
        
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr']
            break
        
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=lr,
            **outputs,
        )
        
        # random resizing
        if (self.progress_in_iter + 1) % 10 == 0:
            self.input_size = self.exp.random_resize(
                self.train_loader, self.epoch, self.rank, self.is_distributed
            )
    
    def before_train(self):
        logger.info("args: {}".format(self.args))
        logger.info("exp value:\n{}".format(self.exp))

        # model related init
        model = self.exp.get_model()
        logger.info(
            "Model Summary: {}".format(get_model_info(model, self.exp.test_size))
        )
        model.cuda()

        # solver related init
        self.optimizer = self.exp.get_optimizer(self.args.batch_size)

        # value of epoch will be set in `resume_train`
        model = self.resume_train(model)

        # data related init
        self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs

        self.train_loader = self.exp.get_data_loader(
            batch_size=self.args.batch_size,
            is_distributed=self.is_distributed,
            no_aug=self.no_aug,
            cache_img=False,
        )

        logger.info("Initializing prefetcher, this might take one minute or less...")
        self.prefetcher = DataPrefetcher(self.train_loader)
        
        if self.args.occupy:
            occupy_mem(0)

        if self.use_model_ema:
            self.ema_model = ModelEMA(model, 0.9998)
            self.ema_model.updates = self.max_iter * self.start_epoch
            
        self.model = model
        self.model.train()
        
        self.face_evaluator, self.body_evaluator = self.exp.get_evaluator(
            batch_size=self.args.batch_size, is_distributed=self.is_distributed
        )

        logger.info("Training start...")
        logger.info("\n{}".format(model))

    
    def after_epoch(self):
        self.save_ckpt("latest", False)

        if (self.epoch + 1) % self.exp.eval_interval == 0:
            all_reduce_norm(self.model)
            self.evaluate_and_save_model()
    
    def before_epoch(self):
        logger.info("---> start train epoch{}".format(self.epoch + 1))

        if self.epoch + 1 == self.exp.l1_loss_start or self.no_aug:
            logger.info("---> No mosaic aug now!")
            self.train_loader.close_mosaic()
            logger.info("---> Add additional L1 loss now!")
            self.model.face_head.loss_fn.use_l1 = True
            self.model.body_head.loss_fn.use_l1 = True
            self.exp.eval_interval = 1