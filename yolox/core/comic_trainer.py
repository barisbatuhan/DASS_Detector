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
)

from .base_trainer import BaseTrainer

class ComicTrainer(BaseTrainer):
    
    def __init__(self, exp, args):
        super().__init__(exp, args)
        self.body_repeat = 2

        
    def train_one_iter(self):
        
        iter_start_time = time.time()
        
        # face update
        if self.exp.head_mode in [0, 1]:
        
            inps, targets = self.face_prefetcher.next()
            inps = inps.to(self.data_type)
            targets = targets.to(self.data_type)
            targets.requires_grad = False
            inps, targets = self.exp.preprocess(inps, targets, self.input_size)
            data_end_time = time.time()
            
            with torch.cuda.amp.autocast(enabled=self.amp_training):

                foutputs = self.model(inps, targets, mode=1, 
                                      boost_ratio=[-1, -1], 
                                      boost_thold=[-1, -1])

            loss = foutputs["total_loss"]
    
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.use_model_ema:
                self.ema_model.update(self.model)

            outputs = {
                "floss": foutputs["total_loss"],
                "fiou_loss": foutputs["iou_loss"],
                "fl1_loss": foutputs["l1_loss"],
                "fconf_loss": foutputs["conf_loss"]
            }

        else:
            outputs = {}
                
        self.model.zero_grad()
        
        # body_update
        if self.exp.head_mode in [0, 2]:
  
            for _ in range(self.body_repeat):
        
                inps, targets = self.body_prefetcher.next()
                inps = inps.to(self.data_type)
                targets = targets.to(self.data_type)
                targets.requires_grad = False
                inps, targets = self.exp.preprocess(inps, targets, self.input_size)
                
                with torch.cuda.amp.autocast(enabled=self.amp_training):
                
                    boutputs = self.model(inps, targets, mode=2,  
                                          boost_ratio=[-1, -1], 
                                          boost_thold=[-1, -1])
            
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
                self.train_face_loader, self.epoch, self.rank, self.is_distributed
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
        if self.args.ckpt is not None:
            model = self.resume_train(model)
        else:
            self.start_epoch = 0

        # data related init
        self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs

        self.train_face_loader, self.train_body_loader = self.exp.get_data_loader(
            batch_size=self.args.batch_size,
            is_distributed=self.is_distributed,
            no_aug=self.no_aug,
            cache_img=False,
        )

        logger.info("Initializing prefetcher, this might take one minute or less...")
        self.face_prefetcher = DataPrefetcher(self.train_face_loader)
        self.body_prefetcher = DataPrefetcher(self.train_body_loader)
        
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

    
    def before_epoch(self):
        logger.info("---> start train epoch{}".format(self.epoch + 1))

        if self.epoch + 1 == self.exp.l1_loss_start or self.no_aug:
            logger.info("---> No mosaic aug now!")
            self.train_face_loader.close_mosaic()
            self.train_body_loader.close_mosaic()
            logger.info("---> Add additional L1 loss now!")
            self.model.face_head.loss_fn.use_l1 = True
            self.model.body_head.loss_fn.use_l1 = True
            self.exp.eval_interval = 1
        
        self.train_face_loader.dataset.arrange_files()
        self.train_body_loader.dataset.arrange_files()