#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch.nn as nn

from .yolo_head import YOLOXHead
from .yolo_head_stem import YOLOXHeadStem
from .yolo_pafpn import YOLOPAFPN


class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head_stem=None, face_head=None, body_head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if face_head is None:
            face_head = YOLOXHead(1)
        if body_head is None:
            body_head = YOLOXHead(1)
        if head_stem is None:
            head_stem = YOLOXHeadStem()

        self.backbone = backbone
        self.face_head = face_head
        self.body_head = body_head
        self.head_stem = head_stem

    # mode = 0 for both heads, 1 for face, 2 for body
    def forward(self, x, targets=None, mode=0, boost_ratio=[-1, -1], boost_thold=[0, 0], unsupervised_train=False):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)
        fpn_outs = self.head_stem(fpn_outs)
        
        if self.training or (mode == 1 and self.face_head.training) or (mode == 2 and self.body_head.training):
            assert targets is not None
            assert mode in [1, 2] # no mode 0 is supported on training
            
            if boost_ratio[0] > 0:
                assert mode == 1
            if boost_ratio[1] > 0:
                assert mode == 2

            outputs = {}
        
            if boost_ratio[0] > 0:
                
                boost_loss = self.body_head(fpn_outs, labels=targets, imgs=x, boost_face=False,
                                            boost_body_from_face_thold=boost_thold[0])
                outputs.update({"total_loss": boost_ratio[0] * boost_loss, "b_bst_loss": boost_loss})
            
            elif boost_ratio[1] > 0:
                boost_loss = self.face_head(fpn_outs, labels=targets, imgs=x, boost_face=True,
                                            boost_face_from_body_thold=boost_thold[1])
                outputs.update({"total_loss": boost_ratio[1] * boost_loss, "f_bst_loss": boost_loss})
            
            elif unsupervised_train:
                
                head = self.face_head if mode == 1 else self.body_head
                loss, conf_loss, reg_loss, num_fg = head(
                    fpn_outs, labels=targets, imgs=x
                )
                outputs.update({
                    "total_loss": loss,
                    "reg_loss": reg_loss,
                    "conf_loss": conf_loss,
                    "num_fg": num_fg,
                })

            else:
                head = self.face_head if mode == 1 else self.body_head
                loss, iou_loss, conf_loss, l1_loss, num_fg = head(
                    fpn_outs, labels=targets, imgs=x
                ) 

                outputs.update({
                    "total_loss": loss,
                    "iou_loss": iou_loss,
                    "l1_loss": l1_loss,
                    "conf_loss": conf_loss,
                    "num_fg": num_fg,
                })

            return outputs
            
        else:
            
            assert mode in [0, 1, 2]
            
            foutputs, boutputs = None, None
            
            if mode in [0, 1]:
                foutputs = self.face_head(fpn_outs)
            if mode in [0, 2]:
                boutputs = self.body_head(fpn_outs)

            return foutputs, boutputs
