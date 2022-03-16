#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from loguru import logger

from yolox.utils import bboxes_iou, intersect, cxcywh2xyxy, check_center_boxa_in_boxb


def get_loss_fn(loss_fn):
    
    loss_dict = {
        "yolox": YOLOXloss,
        "unsupervised": OHEMloss,
    }
    
    assert loss_fn in loss_dict.keys()
    return loss_dict[loss_fn]


class OHEMloss(nn.Module):
    
    def __init__(self, num_neg_ratio=3, 
                 upper_conf_thold=0.8, lower_conf_thold=0.3, 
                 upper_iou_thold=0.5, lower_iou_thold=0.35,
                 reg_loss_coeff=1.0, max_num_objs=10, 
                 random_select=False, **kwargs):
        
        super().__init__()
        self.num_neg_ratio    = num_neg_ratio
        self.upper_conf_thold = upper_conf_thold
        self.lower_conf_thold = lower_conf_thold
        self.upper_iou_thold  = upper_iou_thold
        self.lower_iou_thold  = lower_iou_thold
        self.reg_loss_coeff   = reg_loss_coeff
        self.max_num_objs     = max_num_objs
        self.decode           = True
        
        self.select_neg_boxes_random = random_select
        
        self.bcewithlog_loss  = nn.BCEWithLogitsLoss(reduction="none")
        self.reg_loss         = nn.SmoothL1Loss(reduction="none")
            
    
    def forward(self, labels=None, outputs=None, upper_conf_thold=None, **kwargs):
        
        if upper_conf_thold is not None:
            self.upper_conf_thold  = upper_conf_thold
        
        pos_selected, conf_loss, reg_loss = 0, 0, 0

        for i in range(labels.shape[0]):
            
            obj_indices = torch.zeros_like(outputs[i,:,4])
            label       = labels[i, ...]
            last_idx    = torch.where(label[:,3] == 0)
            
            if (last_idx is None or 
                len(last_idx) < 1 or 
                len(last_idx[0]) < 1 or
                last_idx[0] is None or
                last_idx[0][0] is None):
                last_idx = label.shape[0]
            else:
                last_idx = last_idx[0][0]
            
            if last_idx > self.max_num_objs:
                continue
            
            gt_coords   = label[:last_idx, ...] # [cx, cy, w, h]
            pred_conf   = outputs[i, :, 4].detach().sigmoid()
            pred_coords = outputs[i, :, :4].detach()
            
            ## Selection of Positive and Negative Conf. Indices
            
            thold_upper = torch.where(pred_conf > self.upper_conf_thold)
            
            if len(thold_upper) < 1 or len(thold_upper[0]) < 1:
                continue
            else:
                thold_upper = thold_upper[0]
                
            thold_lower = torch.where(pred_conf < self.lower_conf_thold)
            if len(thold_lower) < 1 or len(thold_lower[0]) < 1:
                continue
            else:
                thold_lower = thold_lower[0]
                
            # IOU Extraction
            
            iou_vals = bboxes_iou(pred_coords, gt_coords, xyxy=False)
            pos_ious, pos_prior_indices = iou_vals[thold_upper, :].max(dim=0)
            neg_ious, _ = iou_vals[thold_lower, :].max(dim=0)

            # Positive IOU Box Selection
            gt_idx = torch.where(pos_ious >= self.upper_iou_thold)[0]
            prior_idx = thold_upper[pos_prior_indices[gt_idx]]
            
            num_poses = prior_idx.shape[0]
            if num_poses == 0:
                continue

            obj_indices[prior_idx] = 1.0 
            
            
            if self.select_neg_boxes_random:
                # Probabilistic Negative IOU Box Selection
                neg_indices    = torch.where(neg_ious <= self.lower_iou_thold)[0]
                neg_indices    = thold_lower[neg_indices]
                neg_indices    = neg_indices[
                    torch.argsort(pred_conf[neg_indices], 
                                  descending=True)[:self.num_neg_ratio * num_poses]
                ]

            else: 
                # RANDOMIZED NEGATIVE BOX SELECTION
                probs = torch.Tensor(
                    [1/max(thold_lower.shape[0], 1)] * thold_lower.shape[0])
                neg_indices = probs.multinomial(
                    num_samples=self.num_neg_ratio * num_poses, replacement=False)
                neg_indices = thold_lower[neg_indices]
            
            
            total_indices = torch.cat([prior_idx, neg_indices]).long()
            
            conf_loss_per_iter = self.bcewithlog_loss(
                outputs[i, total_indices, 4].flatten(),
                obj_indices[total_indices].flatten()
            ).sum()
            
            del total_indices
            
#             if conf_loss_per_iter / num_poses <= 1:
#                 pos_selected += num_poses
#                 conf_loss += conf_loss_per_iter

            pos_selected += num_poses
            conf_loss += conf_loss_per_iter
            
            if self.reg_loss_coeff > 0:
                pos_gt = gt_coords[gt_idx, ...]
                
                max_lens = torch.max(pos_gt[:,2:], dim=1).values.unsqueeze(-1)
                reg_loss += (self.reg_loss(outputs[i, prior_idx, :4], pos_gt) / max_lens).sum()
        
        if pos_selected < 1:
            print("Entered to the no loss area!")
            return 0.0, 0.0, 0.0, 0.0
        else:
            loss = (conf_loss + self.reg_loss_coeff * reg_loss) / pos_selected 
            return loss, conf_loss / pos_selected, reg_loss / pos_selected, pos_selected
            

class YOLOXloss(nn.Module):
    
    def __init__(self, 
                 strides=[8, 16, 32],
                 in_channels=[256, 512, 1024],
                 use_l1=False):
        super().__init__()
        
        self.use_l1 = use_l1
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)
        self.decode = False
        
        
    def forward(self, imgs=None, labels=None, reg_preds=None, obj_preds=None, 
                inp_type=None, **kwargs):
        
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (reg_output, obj_output, stride_this_level) in enumerate(
            zip(reg_preds, obj_preds, self.strides)
        ):

            output = torch.cat([reg_output, obj_output], 1)
            output, grid = self.get_output_and_grid(
                output, k, stride_this_level, inp_type
            )
            x_shifts.append(grid[:, :, 0])
            y_shifts.append(grid[:, :, 1])
            expanded_strides.append(
                torch.zeros(1, grid.shape[1])
                .fill_(stride_this_level)
                .type(inp_type)
            )
            if self.use_l1:
                batch_size = reg_output.shape[0]
                hsize, wsize = reg_output.shape[-2:]
                reg_output = reg_output.view(
                    batch_size, 1, 4, hsize, wsize
                )
                reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                    batch_size, -1, 4
                )
                origin_preds.append(reg_output.clone())
    
            outputs.append(output)
             
        return self.get_losses(
            imgs,
            x_shifts,
            y_shifts,
            expanded_strides,
            labels,
            torch.cat(outputs, 1),
            origin_preds,
            dtype=inp_type)
    
    
    def get_losses(
        self,
        imgs,
        x_shifts,
        y_shifts,
        expanded_strides,
        labels,
        outputs,
        origin_preds,
        dtype,
    ):
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]
               
        # calculate targets
        mixup = labels.shape[2] > 5
        if mixup:
            label_cut = labels[..., :5]
        else:
            label_cut = labels
        nlabel = (label_cut.sum(dim=2) > 0).sum(dim=1)  # number of objects
       
        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)
 
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                gt_classes = labels[batch_idx, :num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]

                try:
                    (
                        fg_mask,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_classes,
                        gt_bboxes_per_image,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                    )
                except RuntimeError:
                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    (
                        fg_mask,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_classes,
                        gt_bboxes_per_image,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                        "cpu",
                    )
            
                torch.cuda.empty_cache()
                num_fg += num_fg_img

                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )
     
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.cuda().float())
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)
  
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)

        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)
        
        loss_iou = (
            self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
        ).sum() / num_fg

        loss_obj = (
            self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
        ).sum() / num_fg

        if self.use_l1:
            loss_l1 = (
                self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
            ).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 5.0

        ### CALCULATION OF THE TOTAL LOSS !!!!!
        loss = reg_weight * loss_iou + loss_obj + loss_l1
        
        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_l1,
            num_fg / max(num_gts, 1),
        )

    def get_output_and_grid(self, output, k, stride, dtype):
        # converted to class part-free
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 5
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid

        output = output.view(batch_size, 1, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, hsize * wsize, -1
        )
        grid = grid.view(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid
    
    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
    def get_assignments(
        self,
        batch_idx,
        num_gt,
        total_num_anchors,
        gt_classes,
        gt_bboxes_per_image,
        bboxes_preds_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        bbox_preds,
        obj_preds,
        labels,
        imgs,
        mode="gpu",
    ):

        if mode == "cpu":
            print("------------CPU Mode for This Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
        )

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)
        gt_obj_per_image = torch.ones(num_gt, num_in_boxes_anchor, 1).cuda()
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            obj_preds_ = obj_preds_.cpu()
            
        # print(obj_preds_.shape, pair_wise_ious_loss.shape)
        
        with torch.cuda.amp.autocast(enabled=False):
            pair_wise_obj_loss = F.binary_cross_entropy_with_logits(
                obj_preds_.unsqueeze(0).repeat(num_gt, 1, 1), 
                gt_obj_per_image, reduction="none"
            ).sum(-1)
        
        del obj_preds_, gt_obj_per_image
        
        cost = (
            pair_wise_obj_loss
            + 3.0 * pair_wise_ious_loss
            + 100000.0 * (~is_in_boxes_and_center)
        )

        (
            num_fg,
            matched_gt_inds,
        ) = self.dynamic_k_matching(cost, pair_wise_ious, num_gt, fg_mask)
        del pair_wise_obj_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            fg_mask = fg_mask.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            fg_mask,
            matched_gt_inds,
            num_fg,
        )

    def get_in_boxes_info(
        self,
        gt_bboxes_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        total_num_anchors,
        num_gt,
    ):
        expanded_strides_per_image = expanded_strides[0]
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image
        x_centers_per_image = (
            (x_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )  # [n_anchor] -> [n_gt, n_anchor]
        y_centers_per_image = (
            (y_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )

        gt_bboxes_per_image_l = (
            (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_r = (
            (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_t = (
            (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_b = (
            (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )

        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
        # in fixed center

        center_radius = 2.5

        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

        is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, num_gt, fg_mask):
        # Dynamic K
        # ---------------------------------------------------------------
        matching_matrix = torch.zeros_like(cost)

        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0.0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
        fg_mask_inboxes = matching_matrix.sum(0) > 0.0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes
        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)

        return num_fg, matched_gt_inds
    
class IOUloss(nn.Module):
    
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        iou = (area_i) / (area_p + area_g - area_i + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_i) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss