#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import datetime
import os
import copy
import time
from loguru import logger

import numpy as np

import torch

from yolox.utils import (
    ModelEMA,
    get_model_info,
    load_ckpt,
    occupy_mem,
    save_checkpoint,
    setup_logger,
    postprocess
)

from .base_trainer import BaseTrainer

class UnsupervisedTrainer(BaseTrainer):
    
    def __init__(self, exp, args):
        super().__init__(exp, args)

    def format_to_target_tensor(self, preds, weak_info, strong_info, orig_size,
                                max_len=120, conf_thold=0.9, nms_thold=0.4):
        
        preds = postprocess(preds, 1, conf_thold, nms_thold)
        target = torch.zeros(len(preds), max_len, 4).cuda()
        convert_annotations = self.prefetcher._dataset.convert_annots_from_weak_to_strong
        
        valid_indices = []
        for i, pred in enumerate(preds):
            
            if pred is None or len(pred) == 0:
                continue
            
            converted_pred = convert_annotations(
                copy.deepcopy(pred[:,:4]), weak_info, strong_info, orig_size, index=i)
            
            if converted_pred is None or len(converted_pred) == 0:
                continue
            
            num_instances = converted_pred.shape[0]
            
            if num_instances > max_len:
                converted_pred = converted_pred[:max_len, ...]
                num_instances = max_len
            
            target[i, :num_instances, :] = converted_pred
            valid_indices.append(i)
            
        if len(valid_indices) == 0:
            return None, None
        
        # if there are none values as targets, these are eliminated
        valid_indices = torch.LongTensor(valid_indices)
        target        = target[valid_indices, ...]
        
        # eliminating negative coordinate predictions
        target[:,:,0] = torch.max(target[:,:,0], torch.zeros(max_len).cuda())
        target[:,:,1] = torch.max(target[:,:,1], torch.zeros(max_len).cuda())
        
        # converting to cxcywh format from x1y1x2y2
        target[:,:,2] = target[:,:,2] - target[:,:,0]
        target[:,:,3] = target[:,:,3] - target[:,:,1]
        target[:,:,0] = target[:,:,0] + target[:,:,2] * 0.5
        target[:,:,1] = target[:,:,1] + target[:,:,3] * 0.5
        
        # If it is turned into supervised training loss, include that
        if self.exp.use_focal_loss:
            target = torch.cat([
                torch.zeros(target.shape[:-1]).to(target.get_device()).unsqueeze(-1), 
                target], dim=-1)
        
        return target, valid_indices

    
    def train_one_iter(self):
        
        iter_start_time = time.time()
        data_end_time = None
        
        for face_iter_idx in range(self.exp.face_num_iter):
        
            # Getting the labels from the teacher model
            weak_inps, heavy_inps, weak_info, heavy_info, orig_size = self.prefetcher.next()
            weak_inps, heavy_inps = weak_inps.cuda(), heavy_inps.cuda()
            
            ### TODO: add preprocess method to run the model with different image sizes.
            
            if face_iter_idx == 0:
                data_end_time = time.time()
            
            with torch.no_grad():
                face_targets, body_targets = self.ema_model.ema(weak_inps, None, mode=0)
    
                face_targets, face_valid_idx = self.format_to_target_tensor(
                    face_targets, copy.deepcopy(weak_info), copy.deepcopy(heavy_info), 
                    orig_size, conf_thold=self.exp.teacher_face_conf_thold,
                    nms_thold=self.exp.teacher_face_nms_thold)
                
            trained_face = False
            if face_valid_idx is not None and len(face_valid_idx) > 0 and self.exp.head_mode in [0, 1]:
                trained_face = True
    
                with torch.cuda.amp.autocast(enabled=self.amp_training):
                    foutputs = self.model(copy.deepcopy(heavy_inps[face_valid_idx, ...]), 
                                          copy.deepcopy(face_targets), unsupervised_train=not self.exp.use_focal_loss,
                                          mode=1)
    
                loss = foutputs["total_loss"]
                
                if type(loss) != float:
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
        
        self.model.zero_grad()
        
        for body_iter_idx in range(self.exp.body_num_iter):
            
            # Getting the labels from the teacher model
            weak_inps, heavy_inps, weak_info, heavy_info, orig_size = self.prefetcher.next()
            weak_inps, heavy_inps = weak_inps.cuda(), heavy_inps.cuda()
            
            ### TODO: add preprocess method to run the model with different image sizes.
            
            if data_end_time is None and body_iter_idx == 0:
                data_end_time = time.time()
            
            with torch.no_grad():
                face_targets, body_targets = self.ema_model.ema(weak_inps, None, mode=0)
                
                body_targets, body_valid_idx = self.format_to_target_tensor(
                    body_targets, weak_info, heavy_info, orig_size,
                    conf_thold=self.exp.teacher_body_conf_thold,
                    nms_thold=self.exp.teacher_body_nms_thold)
        
        
            trained_body = False
            if body_valid_idx is not None and len(body_valid_idx) > 0 and self.exp.head_mode in [0, 2]:
                trained_body = True
            
                with torch.cuda.amp.autocast(enabled=self.amp_training):
                    boutputs = self.model(copy.deepcopy(heavy_inps[body_valid_idx, ...]), 
                                          copy.deepcopy(body_targets), unsupervised_train=not self.exp.use_focal_loss,
                                          mode=2)
    
                loss = boutputs["total_loss"]
                
                if type(loss) != float:
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
           
        outputs = {}

        if trained_face:
            outputs.update({
                "floss": foutputs["total_loss"],
                "freg_loss": foutputs["reg_loss"] if not self.exp.use_focal_loss else foutputs["iou_loss"],
                "fconf_loss": foutputs["conf_loss"],
                "fnum_fg": foutputs["num_fg"],
            })
        
        if trained_body:
            outputs.update({
                "bloss": boutputs["total_loss"],
                "breg_loss": boutputs["reg_loss"] if not self.exp.use_focal_loss else boutputs["iou_loss"],
                "bconf_loss": boutputs["conf_loss"],
                "bnum_fg": boutputs["num_fg"],
            })
        
        if trained_face or trained_body:
            
            self.exp.change_conf_tholds()
        
            if (self.iter > 0 
                and self.iter % self.exp.update_teacher_per_iter == 0
                and self.use_model_ema):
                self.ema_model.update(self.model)

            for param_group in self.optimizer.param_groups:
                lr = param_group["lr"]
                break
            
            iter_end_time = time.time()

            self.meter.update(
                iter_time=iter_end_time - iter_start_time,
                data_time=data_end_time - iter_start_time,
                lr=lr,
                **outputs
            )
        
        if (self.exp.match_models_per_iter > 0 
            and self.progress_in_iter > 0 
            and self.progress_in_iter % self.exp.match_models_per_iter == 0
            and self.use_model_ema):
            
            self.model.load_state_dict(copy.deepcopy(self.ema_model.ema.state_dict()))
            self.model.train()
            self.ema_model.updates = 0
            
    
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
        model, ema_model = self.resume_train(model)
        logger.info("Models are loaded!")

        # data related init
        self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs
        self.train_loader = self.exp.get_data_loader(
            batch_size=self.args.batch_size,
            is_distributed=self.is_distributed,
            no_aug=self.no_aug,
            cache_img=False
        )
        
        logger.info("Constructing the Data Loader Iterator..")
        self.prefetcher = iter(self.train_loader)
        
        if self.args.occupy:
            logger.info("Occupying memory since it is enabled...")
            occupy_mem(0)
        
        self.model = model
        self.model.train() 
            
        logger.info("Applying final settings on the teacher model...")

        # Definition of the Teacher Model
        self.ema_model = ModelEMA(ema_model, self.exp.ema_keep_rate, 
                                  ema_exp_denominator=self.exp.ema_exp_denominator,
                                  const_rate=self.exp.const_ema_rate,
                                  reverse=self.exp.reverse)
        
        if self.exp.match_models_per_iter > 0 or self.exp.const_ema_rate: 
            self.ema_model.updates = 0 
        else:
            self.ema_model.updates = self.max_iter * self.start_epoch
        
        logger.info("Creating Evaluators...")        

        self.face_evaluator, self.body_evaluator = self.exp.get_evaluator(
            batch_size=self.args.batch_size, is_distributed=self.is_distributed
        )

        logger.info("Training starts...")
        logger.info("\n{}".format(model))
        
        # self.evaluate_and_save_model()


    def before_epoch(self):
        logger.info("---> start train epoch {}".format(self.epoch + 1))
        
        self.model.train()
        self.ema_model.ema.eval()

        if self.epoch + 1 == self.exp.l1_loss_start or self.no_aug:
            logger.info("---> Add additional L1 loss now!")
            self.model.face_head.use_l1 = True
            self.model.body_head.use_l1 = True
            self.exp.eval_interval = 1


    def resume_train(self, model):
        
        if self.args.resume:
            ckpt_file = os.path.join(
                "YOLOX_outputs/", self.args.experiment_name, "latest_ckpt.pth")
            
            if os.path.isfile(ckpt_file):
                self.args.ckpt = ckpt_file
        
        assert self.args.ckpt is not None
        
        ckpt = torch.load(self.args.ckpt, map_location=self.device)
        
        if "student_model" in ckpt and "teacher_model" in ckpt:
            model.load_state_dict(ckpt["teacher_model"])
            ema_model = copy.deepcopy(model)
            ema_model.load_state_dict(ckpt["teacher_model"])
        
        else:
            model.load_state_dict(ckpt["model"])
            ema_model = copy.deepcopy(model)

        if "best_body_val" in ckpt:
            self.best_body_ap = ckpt["best_body_val"]
        if "best_face_val" in ckpt:
            self.best_face_ap = ckpt["best_face_val"]
        if "best_avg_val" in ckpt:
            self.best_avg_ap = ckpt["best_avg_val"]
        
        
        if self.args.resume and "latest" in self.args.ckpt:
            
            self.optimizer.load_state_dict(ckpt["optimizer"])
            
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
            logger.info("Training from epoch 1 - Loaded checkpoint '{}'".format(
                self.args.ckpt))
            
        return model, ema_model
    
    
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
            better_results = True
            self.best_face_ap = max(self.best_face_ap, fmAP)
            self.save_ckpt("best_face_model", False)
        
        if bmAP > self.best_body_ap and self.exp.head_mode in [0, 2]:
            self.best_body_ap = max(self.best_body_ap, bmAP)
            self.save_ckpt("best_body_model", False)
        
        if self.exp.head_mode == 0 and (bmAP + fmAP) / 2 > self.best_avg_ap:
            self.best_avg_ap = max(self.best_avg_ap, (bmAP + fmAP) / 2)
            self.save_ckpt("best_avg_model", False)
            
        

    def save_ckpt(self, ckpt_name, update_best_ckpt=False):
        if self.rank == 0:
            # save_model = self.ema_model.ema if self.use_model_ema else self.model
            logger.info("Save weights to {}".format(self.file_name))
            ckpt_state = {
                "start_epoch": self.epoch + 1,
                "teacher_model": self.ema_model.ema.state_dict(),
                "student_model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "best_body_val": self.best_body_ap,
                "best_face_val": self.best_face_ap,
                "best_avg_val": self.best_avg_ap,
            }
            save_checkpoint(
                ckpt_state,
                update_best_ckpt,
                self.file_name,
                ckpt_name,
            )
