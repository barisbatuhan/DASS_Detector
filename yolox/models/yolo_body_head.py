#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import math
import time
from loguru import logger

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolox.utils import bboxes_iou, intersect, cxcywh2xyxy

from .losses import IOUloss, BasicBoostloss, GreedyBoostloss
from .network_blocks import BaseConv, DWConv
from .yolo_head import YOLOXHead


class YOLOXBodyHead(YOLOXHead):
    def __init__(
        self,
        num_classes,
        width=1.0,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
        boost_type=None
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__(num_classes, width, strides, in_channels, act, depthwise)
        
        if boost_type == None:
            self.boost_loss = None
        elif boost_type == "basic":
            self.boost_loss = BasicBoostloss()
        elif boost_type == "greedy":
            self.boost_loss = GreedyBoostLoss()
        else:
            raise NotImplementedError

    def set_boost_loss(self, boost_type):
        if boost_type == None:
            self.boost_loss = None
        elif boost_type == "basic":
            self.boost_loss = BasicBoostloss()
        elif boost_type == "greedy":
            self.boost_loss = GreedyBoostLoss()
        else:
            raise NotImplementedError
    
    
    def forward(self, xin, labels=None, imgs=None, boost_body_from_face_thold=0):
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (reg_conv, stride_this_level, x) in enumerate(
            zip(self.reg_convs, self.strides, xin)
        ):
            reg_feat = reg_conv(x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            if self.training and not boost_body_from_face_thold > 0:
                output = torch.cat([reg_output, obj_output], 1)
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, xin[0].type()
                )
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1])
                    .fill_(stride_this_level)
                    .type_as(xin[0])
                )
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(
                        batch_size, self.n_anchors, 4, hsize, wsize
                    )
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4
                    )
                    origin_preds.append(reg_output.clone())

            elif self.training and boost_body_from_face_thold > 0:
                output = torch.cat([reg_output, obj_output], 1)  
            else:
                output = torch.cat([reg_output, obj_output.sigmoid()], 1)

            outputs.append(output)
        
        if self.training and not boost_body_from_face_thold > 0:
            return self.get_losses(
                imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                torch.cat(outputs, 1),
                origin_preds,
                dtype=xin[0].dtype,
            )
        elif self.training and boost_body_from_face_thold > 0:

            self.hw = [x.shape[-2:] for x in outputs]
            
            outputs = torch.cat(
                    [x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)
            
            if self.decode_in_inference:
                outputs = self.decode_outputs(outputs, dtype=xin[0].type())
            
            return self.boost_loss(labels, outputs, 
                                   boost_body_from_face_thold, boost_face=False)

        else:
            self.hw = [x.shape[-2:] for x in outputs]
            # [batch, n_anchors_all, 85]
            outputs = torch.cat(
                [x.flatten(start_dim=2) for x in outputs], dim=2
            ).permute(0, 2, 1)
            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=xin[0].type())
            else:
                return outputs