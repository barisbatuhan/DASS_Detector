#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import datetime
import os
import time
from loguru import logger
import traceback

from abc import ABCMeta, abstractmethod

import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

from yolox.data import DataPrefetcher
from yolox.utils import (
    MeterBuffer,
    ModelEMA,
    all_reduce_norm,
    get_local_rank,
    get_model_info,
    get_rank,
    get_world_size,
    gpu_mem_usage,
    is_parallel,
    load_ckpt,
    occupy_mem,
    save_checkpoint,
    setup_logger,
    synchronize
)


class BaseTrainer:
    def __init__(self, exp, args):
        # init function only defines some basic attr, other attrs like model, 
        # optimizer are built in before_train methods.
        self.exp = exp
        self.args = args

        # training related attr
        self.max_epoch = exp.max_epoch
        self.amp_training = args.fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
        self.use_model_ema = exp.ema

        # data/dataloader related attr
        self.data_type = torch.float16 if args.fp16 else torch.float32
        self.input_size = exp.input_size
        self.best_face_ap = 1e-8
        self.best_body_ap = 1e-8
        self.best_avg_ap  = 1e-8

        # metric record
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.file_name = os.path.join(exp.output_dir, args.experiment_name)

        # only call in rank 0 if distributed parallel is also implemented
        self.rank = 0
        self.device = "cuda:0"
        self.is_distributed=False
        os.makedirs(self.file_name, exist_ok=True)

        self.max_iter = self.exp.max_iter
        
        setup_logger(
            self.file_name,
            distributed_rank=self.rank,
            filename="train_log.txt",
            mode="a",
        )

    def train(self):
        try:
            self.before_train()
            self.train_in_epoch()
        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error(e)
            raise
        finally:
            self.after_train()

    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def train_in_iter(self):
        for self.iter in range(self.max_iter):
            self.before_iter()
            self.train_one_iter()
            self.after_iter()

    @abstractmethod
    def train_one_iter(self):
        pass
    
    @abstractmethod
    def before_train(self):
        pass

    def after_train(self):
        logger.info(
            (f'Training of experiment is done and '
             f'the best Face AP: {self.best_face_ap * 100} and '
             f'the best Body AP: {self.best_body_ap * 100}')
        )

    @abstractmethod
    def before_epoch(self):
        pass


    def after_epoch(self):
        # self.save_ckpt("epoch_" + str(self.epoch + 1), False)
        self.save_ckpt("latest", False)

        if (self.epoch + 1) % self.exp.eval_interval == 0:
            all_reduce_norm(self.model)
            self.evaluate_and_save_model()

    
    def before_iter(self):
        pass

    def after_iter(self, print_with_interval=True):
        """
        `after_iter` contains two parts of logic:
            * log information
            * reset setting of resize
        """
        # log needed information
        
        if print_with_interval:
            print_check = (self.iter + 1) % self.exp.print_interval == 0
        else:
            loss_meter = self.meter.get_filtered_meter("loss")
            if (("floss" in loss_meter and loss_meter["floss"].latest is not None and loss_meter["floss"].latest > 0) or 
                ("bloss" in loss_meter and loss_meter["bloss"].latest is not None and loss_meter["bloss"].latest > 0)):
                print_check = True
            else:
                print_check = False
        
        if print_check:

            left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
            eta_seconds = self.meter["iter_time"].global_avg * left_iters
            eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))
            
            progress_str = "epoch: {}/{}, iter: {}/{}".format(
                self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter
            )
            loss_meter = self.meter.get_filtered_meter("loss")
            
            loss_str = ", ".join(
                ["{}: {:.3f}".format(k, 0.0 if v.latest is None else v.latest) for k, v in loss_meter.items()]
            )

            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(
                ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
            )
            
            lr = self.meter["lr"].latest
            lr = 0.0 if lr is None else lr
            
            logger.info(
                "{}, mem: {:.0f}Mb, {}, {}, lr: {:.3e}".format(
                    progress_str,
                    gpu_mem_usage(),
                    time_str,
                    loss_str,
                    lr,
                )
                + (", size: {:s}, {}".format(str(self.input_size[0]), eta_str))
            )
            
            self.meter.clear_meters()

    
    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter

    
    def resume_train(self, model):
        
        if self.args.ckpt is None and self.args.resume:
            ckpt_file = os.path.join("YOLOX_outputs/", 
                                     self.args.experiment_name, 
                                     "latest_ckpt.pth")
        
        # TODO: remove elif statement after the custom training
#         elif os.path.isfile(os.path.join("YOLOX_outputs/", 
#                                          self.args.experiment_name, 
#                                          "latest_ckpt.pth")):
            
#             ckpt_file = os.path.join("YOLOX_outputs/", 
#                                      self.args.experiment_name, 
#                                      "latest_ckpt.pth")
        
        else:
            ckpt_file = self.args.ckpt
              
        if os.path.isfile(ckpt_file) and not self.args.resume:
            ckpt = torch.load(ckpt_file, map_location=self.device)
            
            if "model" in ckpt:
                model.load_state_dict(ckpt["model"])
            elif "teacher_model" in ckpt:
                model.load_state_dict(ckpt["teacher_model"])
            else:
                raise ValueError("No model key found in the pretrained weights file!")
                
            if "best_body_val" in ckpt:
                self.best_body_ap = ckpt["best_body_val"]
            if "best_face_val" in ckpt:
                self.best_face_ap = ckpt["best_face_val"]
            
            # self.optimizer.load_state_dict(ckpt["optimizer"])
            self.start_epoch = 0

            logger.info(
                "Starting Training from Pretrained Weights - Loaded '{}' (epoch {})".format(
                    self.args.ckpt, self.start_epoch + 1
                )
            )
            
        elif (self.args.resume or "latest_ckpt.pth" in ckpt_file) and os.path.isfile(ckpt_file):
            ckpt = torch.load(ckpt_file, map_location=self.device)
            
            if "model" in ckpt:
                model.load_state_dict(ckpt["model"])
            elif "teacher_model" in ckpt:
                model.load_state_dict(ckpt["teacher_model"])
            else:
                raise ValueError("No model key found in the pretrained weights file!")
            
            self.optimizer.load_state_dict(ckpt["optimizer"])
            
            if "best_body_val" in ckpt:
                self.best_body_ap = ckpt["best_body_val"]
            if "best_face_val" in ckpt:
                self.best_face_ap = ckpt["best_face_val"]
            
            # resume the training states variables
            start_epoch = (
                self.args.start_epoch - 1
                if self.args.start_epoch is not None
                else ckpt["start_epoch"]
            )
            self.start_epoch = start_epoch
            logger.info(
                "Resuming Training - Loaded checkpoint '{}' (epoch {})".format(
                    self.args.ckpt, self.start_epoch
                )
            )
        
        else:
            self.start_epoch = 0
            logger.info("Training from epoch 1 - Found no checkpoint!")

            
        return model

    def evaluate_and_save_model(self):
        if self.use_model_ema:
            evalmodel = self.ema_model.ema
        else:
            evalmodel = self.model
            if is_parallel(evalmodel):
                evalmodel = evalmodel.module

        if self.exp.head_mode in [0, 1]:
            face_result = self.exp.eval(
                evalmodel, self.face_evaluator, self.is_distributed, mode=1,
            )
            
            if face_result is None:
                fmAP = 0
            else:
                faps = face_result['ap']
                faps = faps[faps > 0]
                fmAP = faps[faps > 0][0]
                logger.info('-> Face Evaluated # mAP: ' + str(fmAP) + " / " + str(self.best_face_ap))
        
        else:
            fmAP = 0
        
        if self.exp.head_mode in [0, 2]:
            body_result = self.exp.eval(
                evalmodel, self.body_evaluator, self.is_distributed, mode=2)
            
            if body_result is None:
                bmAP = 0
            else:
                baps = body_result['ap']
                baps = baps[baps > 0]
                bmAP = baps[baps > 0][0]
                logger.info('-> Body Evaluated # mAP: ' + str(bmAP) + " / " + str(self.best_body_ap))
        else:
            bmAP = 0
        
        self.model.train()

        if fmAP > self.best_face_ap and self.exp.head_mode in [0, 1]:
            self.best_face_ap = max(self.best_face_ap, fmAP)
            self.save_ckpt("best_face_model", False)
        if bmAP > self.best_body_ap and self.exp.head_mode in [0, 2]:
            self.best_body_ap = max(self.best_body_ap, bmAP)
            self.save_ckpt("best_body_model", False)


    def save_ckpt(self, ckpt_name, update_best_ckpt=False):
        if self.rank == 0:
            save_model = self.ema_model.ema if self.use_model_ema else self.model
            logger.info("Save weights to {}".format(self.file_name))
            ckpt_state = {
                "start_epoch": self.epoch + 1,
                "model": save_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "best_body_val": self.best_body_ap,
                "best_face_val": self.best_face_ap
            }
            save_checkpoint(
                ckpt_state,
                update_best_ckpt,
                self.file_name,
                ckpt_name,
            )