#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
from loguru import logger

import cv2
import numpy as np
from pycocotools.coco import COCO

from ..dataloading import get_yolox_datadir
from .datasets_wrapper import Dataset
from ..data_augment import draw_speech_balloon


class COCOPersonsDataset(Dataset):
    """
    COCO dataset class.
    """

    def __init__(
        self,
        data_dir=None,
        json_file="instances_train2017.json",
        name="train2017",
        img_size=(416, 416),
        preproc=None,
        filter_file=None,
        random_style="all",
        filter_obj_style="living",
        cache=False,
        **kwargs
    ):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        super().__init__(img_size)
        if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), "COCO")
        self.data_dir = data_dir
        self.json_file = json_file
        
        self.set_style(random_style)

        self.coco = COCO(os.path.join(self.data_dir, "annotations", self.json_file))
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in cats])

        self.set_allowed_obj_ids(filter_obj_style)
        
        self.imgs = None
        self.name = name
        self.img_size = img_size
        self.preproc = preproc
        self.annotations = self._load_coco_annotations()
        self.len_ratio = None
        
        if "train" in name and filter_file is not None:
            f = open(filter_file, "r")
            lines = f.readlines()
            
            indices = []
            for line in lines:
                
                if len(line) < 2:
                    continue
                
                elif line[-1] == "\n":
                    line = line[:-1]
                
                index, id_ = line.split(",")
                index, id_ = int(index), int(id_)
                assert self.ids[index] == id_
                indices.append(index)
            
            indices = np.asarray(indices, dtype=int)
            
            self.ids = np.asarray(self.ids)[indices.astype(int)]
            self.annotations = np.asarray(self.annotations, dtype=object)[indices.astype(int)]
            

    def set_style(self, random_style):
        if random_style == "best_comb":
            self.random_style = [
                "cartoongan_hosoda", "cartoongan_hayao", 
                "cartoongan_shinkai", "ganilla_KH", "whitebox"
            ]
        else:
            self.random_style = random_style
    
    def set_allowed_obj_ids(self, filter_obj_style):
        
        all_living_objs = ["person", "bird", "cat", "dog", "horse", "sheep", 
                           "cow", "elephant", "bear", "zebra", "giraffe"]
        
        if filter_obj_style == "all":
            self.allowed_objs = self._classes
        elif filter_obj_style == "living":
            self.allowed_objs = all_living_objs
        elif filter_obj_style == "person":
            self.allowed_objs = ["person"]
        elif filter_obj_style == "nobird":
            self.allowed_objs = ["person"] + all_living_objs[2:] 
        else:
            raise ValueError("Unexpected object filtering style for COCO!")
        
        self.allowed_obj_ids = []
        for i, cls in enumerate(self._classes):
            if cls in self.allowed_objs:
                self.allowed_obj_ids.append(i)
    
    
    def __len__(self):
        return len(self.ids)

    def __del__(self):
        del self.imgs

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                if self.class_ids.index(obj["category_id"]) in self.allowed_obj_ids:
                    obj["clean_bbox"] = [x1, y1, x2, y2]
                    objs.append(obj)

        num_objs = len(objs)

        res = np.zeros((num_objs, 5))

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))

        file_name = (
            im_ann["file_name"]
            if "file_name" in im_ann
            else "{:012}".format(id_) + ".jpg"
        )

        return (res, img_info, resized_info, file_name)

    def load_anno(self, index):
        return self.annotations[index][0]

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def load_image(self, index):
        file_name = self.annotations[index][3]
        orig_img_file = None
        if "train" in self.name:
            if self.random_style == "all":
                different_styles = os.listdir(os.path.join(self.data_dir, self.name))
                random_style = different_styles[np.random.randint(len(different_styles))]
            elif type(self.random_style) == str:
                random_style = self.random_style
            else:
                random_style = np.random.choice(self.random_style)
            
            img_file = os.path.join(self.data_dir, self.name, random_style, file_name)
            orig_img_file = os.path.join(self.data_dir, self.name, "original", file_name)
        
        else:
            img_file = os.path.join(self.data_dir, self.name, file_name)
        
        
        # print(img_file)
        img = cv2.imread(img_file)
        assert img is not None
        
        if orig_img_file is not None:
            orig_img = cv2.imread(orig_img_file, cv2.IMREAD_COLOR)
            if img.shape != orig_img.shape:
                h, w, c = orig_img.shape
                img = cv2.resize(img, [w, h]) 

        return img

    def pull_item(self, index):
        id_ = self.ids[index]

        res, img_info, resized_info, _ = self.annotations[index]
        
        if self.len_ratio is not None:
            lens = (res[:,2:4] - res[:,0:2]) / img_info
            selected_indices = np.where(lens > self.len_ratio)[0]
            res = res[selected_indices, :]

        if self.imgs is not None:
            pad_img = self.imgs[index]
            img = pad_img[: resized_info[0], : resized_info[1], :].copy()
        else:
            img = self.load_resized_img(index)

        if self.len_ratio is not None:
            selected_indices = np.asarray(
                [i for i, val in enumerate(res[:,-1]) if val in self.obj_indices])
            res = res[selected_indices, :]
        
        return img, res.copy(), img_info, np.array([id_])

    @Dataset.mosaic_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w.
                h, w (int): original shape of the image
            img_id (int): same as the input index. Used for evaluation.
        """
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        return img, target, img_info, img_id
