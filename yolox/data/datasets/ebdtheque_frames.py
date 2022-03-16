import os
from collections import OrderedDict
from loguru import logger

import cv2
import numpy as np

import json

from ..dataloading import get_yolox_datadir
from .datasets_wrapper import Dataset


class EBDthequeFramesDataset(Dataset):
    """
    eBDtheque dataset class.
    """

    def __init__(
        self,
        data_dir=None,
        train=True,
        img_size=(416, 416),
        preproc=None,
        cache=False, # no cache is supported currently
    ):
        super().__init__(img_size)
        self.class_ids = [0]
        self.class_dict = {"body" : self.class_ids[0]}
        self.img_size = img_size
        self.preproc = preproc   
        self.files, self.annotations = self.load_annotations(data_dir, train)
        
        
    def __len__(self):
        return len(self.files)
    
    
    def pull_item(self, index):
        img, img_info = self.load_image(index)
        res = self.load_anno(index)
        return img, res.copy(), img_info, np.array([index])
    
    
    def __getitem__(self, index):
        img, target, img_info, ids = self.pull_item(index)
        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        return img, target, img_info, ids
    
    
    def load_anno(self, index):
        # given an index, it loads the annotations of the file at that index
        file = self.files[index]
        annots = self.annotations[file]
        anno = np.zeros((len(annots), 5))

        for idx, ann in enumerate(annots):
            anno[idx,:4] = ann
            anno[idx,4] = self.class_dict["body"]
        
        return anno
    
    
    def load_resized_img(self, index):
        img, img_info = self.load_image(index)
        
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)

        return resized_img, img_info

    def load_image(self, index):
        img_path = self.files[index]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        assert img is not None
        h, w, c = img.shape
        return img, [h, w]
    
    
    def load_annotations(self, edb_paths, train :bool):
        # gven a path and partition, it loads all the image paths and annots in that partition
        files = []
        boxes = {}
        
        img_path, annot_path = edb_paths["imgs"], edb_paths["labels"]
        
        if annot_path is not None:
            with open(annot_path, "r") as f:
                ann = json.load(f)
            
            for file in ann.keys():
                k = os.path.join(img_path, file)
                files.append(k)
                boxes[k] = ann[file]["body"]
        else:
            files = [os.path.join(iimg_path, file) for file in os.listdir(img_path)]
            for file in files:
                boxes[file] = None

        return files, boxes
        