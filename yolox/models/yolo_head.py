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

from .network_blocks import BaseConv, DWConv


class YOLOXHead(nn.Module):
    def __init__(
        self,
        num_classes,
        width=1.0,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
        loss_fn=None
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()

        self.num_classes = num_classes
        
        self.reg_convs = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        
        Conv = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):

            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

        self.strides = strides
        self.in_channels=in_channels
        self.loss_fn = loss_fn

    def initialize_biases(self, prior_prob):
        for conv in self.obj_preds:
            b = conv.bias.view(1, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xin, labels=None, imgs=None, boost_thold=0, boost_face=False):
        
        reg_outputs, obj_outputs  = [], []
        
        for k, (reg_conv, x) in enumerate(zip(self.reg_convs, xin)):
            reg_feat = reg_conv(x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)
            
            reg_outputs.append(reg_output)
            obj_outputs.append(obj_output)

        if self.training:
            outputs = []

            if self.loss_fn.decode:
                for reg_output, obj_output in zip(reg_outputs, obj_outputs):
                    output = torch.cat([reg_output, obj_output], 1)
                    outputs.append(output)
                
                self.hw = [x.shape[-2:] for x in outputs]
                outputs = torch.cat(
                    [x.flatten(start_dim=2) for x in outputs], dim=2
                ).permute(0, 2, 1)
                
                outputs = self.decode_outputs(outputs, dtype=xin[0].type())
            
            return self.loss_fn(imgs=imgs, labels=labels, 
                                reg_preds=reg_outputs, 
                                obj_preds=obj_outputs,
                                outputs=outputs, 
                                boost_thold=boost_thold, 
                                boost_face=boost_face,
                                inp_type=xin[0].type())
        
        else:
            
            outputs = []
            for reg_output, obj_output in zip(reg_outputs, obj_outputs):
                output = torch.cat([reg_output, obj_output.sigmoid()], 1)
                outputs.append(output)
                
            self.hw = [x.shape[-2:] for x in outputs]
                
            outputs = torch.cat(
                [x.flatten(start_dim=2) for x in outputs], dim=2
            ).permute(0, 2, 1)
            
            outputs = self.decode_outputs(outputs, dtype=xin[0].type())
            
            return outputs
        
        
    def decode_outputs(self, outputs, dtype):
        # converted to class part-free
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs