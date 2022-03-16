import os
from collections import OrderedDict
from loguru import logger

import cv2
import copy
import json
import random
import numpy as np
import torch

from ..dataloading import get_yolox_datadir
from .datasets_wrapper import Dataset


class RawUnsupervisedComicsDataset(Dataset):

    def __init__(
        self,
        data_dir=None,
        train=True,
        weak_img_size=(1024, 1024),
        strong_img_size=(640, 640),
        weak_preproc=None,
        strong_preproc=None,
        cache=False, # no cache is supported currently
    ):
        super().__init__(strong_img_size)
        
        self.weak_img_size = weak_img_size
        self.strong_img_size = strong_img_size
        self.weak_preproc = weak_preproc   
        self.strong_preproc = strong_preproc
        self.files = data_dir
        
    def __len__(self):
        return 50000 # some not so big random value since the choice is purely random
    
    
    def pull_item(self, index):
        img, img_info, min_crop_ratio = self.load_image(index)
        return img, img_info, min_crop_ratio 
    
    
    def __getitem__(self, index):
        img, img_info, min_crop_ratio = self.pull_item(index)
        
        if self.strong_preproc is not None:
            self.strong_preproc.min_crop_ratio = min_crop_ratio 
            strong_img, strong_aug_info = self.strong_preproc(copy.deepcopy(img), self.strong_img_size)
        if self.weak_preproc is not None:
            weak_img, weak_aug_info = self.weak_preproc(img, self.weak_img_size)
        
        return weak_img, strong_img, weak_aug_info, strong_aug_info, img_info 
    
    def convert_annots_from_weak_to_strong(self, boxes, 
                                           init_weak_aug_info, init_strong_aug_info, 
                                           init_orig_size, index=None):
        """
        - Assumes that the boxes array shape will be (N, 4) and the coordinates will be 
        given as x1, y1, x2, y2.
        - Keys of weak_aug_info   --> ratio :float, hflip :bool
        - Keys of strong_aug_info --> ratio :float, hflip :bool, crop : {length :int, x :int, y:int}
        - original size of the image in the format of: h, w
        x and y are the x1 and y1 points that shows the topleft corner of the crop-box
        - To convert to original shape, divide with the ratio.
        """
        if index is not None:
            # when batch input is passed, this if statement handles the case
            weak_aug_info, strong_aug_info, orig_size = {}, {}, [init_orig_size[0][index], init_orig_size[1][index]]
            
            for k in init_weak_aug_info.keys():
                weak_aug_info[k] = init_weak_aug_info[k][index]
            
            for k in init_strong_aug_info.keys():
                if type(init_strong_aug_info[k]) == dict:
                    strong_aug_info[k] = {}
                    for k2 in init_strong_aug_info[k].keys():
                        strong_aug_info[k][k2] = init_strong_aug_info[k][k2][index]
                else:
                    strong_aug_info[k] = init_strong_aug_info[k][index]
            
        else:
            weak_aug_info, strong_aug_info, orig_size = init_weak_aug_info, init_strong_aug_info, init_orig_size,
        
        if type(boxes) == np.ndarray:
            where = np.where
            unique = np.unique
            concat = np.concatenate
            maximum = np.maximum
            minimum = np.minimum
            intersect = np.intersect1d
        else:
            where = torch.where
            unique = torch.unique
            concat = torch.cat
            maximum = torch.maximum
            minimum = torch.minimum
            
            def intersect(t1, t2):
                combined = torch.cat((t1.unique(), t2.unique()))
                uniques, counts = combined.unique(return_counts=True)
                difference = uniques[counts == 1]
                intersection = uniques[counts > 1]
                return intersection
        
        orig_h, orig_w = orig_size
        max_len        = max(orig_size)
        
        # Convert to original form of the annotation
        boxes    = boxes / weak_aug_info["ratio"]
        
        # remove boxes that exceeds the original w and h
        valid_xs = where(boxes[:,0] <= orig_w)[0]
        valid_ys = where(boxes[:,1] <= orig_h)[0]
        valids   = intersect(valid_xs, valid_ys)
        
        if len(valids) < 1:
            # print("Returned none from original w and h setting.")
            return None
        
        boxes    = boxes[valids, :]
            
        # fit the annotation to the original image shape
        if type(boxes) == np.ndarray:
            zero_val = 0 
            maxw_val = orig_w - 1
            maxh_val = orig_h - 1
        else:
            zero_val = torch.zeros_like(boxes[:,:2]) 
            maxw_val = torch.zeros_like(boxes[:,2]).fill_(orig_w - 1)
            maxh_val = torch.zeros_like(boxes[:,3]).fill_(orig_h - 1)
        
        boxes[:,:2] = maximum(boxes[:,:2], zero_val)
        boxes[:,2]  = minimum(boxes[:,2], maxw_val)
        boxes[:,3]  = minimum(boxes[:,3], maxh_val)
        
        # do a horizontal flip if the state in both strong and weak
        # augmentation are not the same
        if weak_aug_info["hflip"] != strong_aug_info["hflip"]:
            boxes[:,0] = orig_w - boxes[:,0]
            boxes[:,2] = orig_w - boxes[:,2]
            
            # swap
            temp        = copy.deepcopy(boxes[:,0])
            boxes[:,0] = boxes[:,2]
            boxes[:,2] = temp
        
        if strong_aug_info["vflip"]:
            temp        = copy.deepcopy(boxes[:,1])
            boxes[:, 1] = orig_h - boxes[:, 3]
            boxes[:, 3] = orig_h - temp
        
        # process the cropping operation in the strong augmentation
        if strong_aug_info["crop"] is not None:
            side_len = strong_aug_info["crop"]["length"]
            x, y     = strong_aug_info["crop"]["x"], strong_aug_info["crop"]["y"] 
            
            valid_x1s = where(boxes[:,2] > x)[0]
            valid_y1s = where(boxes[:,3] > y)[0]
            valid_x2s = where(boxes[:,0] < x + side_len)[0]
            valid_y2s = where(boxes[:,1] < y + side_len)[0]
            
            valids = intersect(intersect(valid_x1s, valid_x2s), 
                               intersect(valid_y1s, valid_y2s))
            
            if len(valids) < 1:
                # print("Returned none from strong aug crop process.")
                # print("Boxes:", boxes)
                return None
            
            boxes  = boxes[valids, :]
            
            if type(boxes) == np.ndarray:
                min_val  = 0 
                maxw_val = side_len - 1
                maxh_val = side_len - 1
            else:
                min_val  = torch.zeros_like(boxes[:,1]) 
                maxw_val = torch.zeros_like(boxes[:,2]).fill_(side_len-1)
                maxh_val = torch.zeros_like(boxes[:,3]).fill_(side_len-1)
        
            boxes[:,0] = maximum(boxes[:,0] - x, min_val)
            boxes[:,1] = maximum(boxes[:,1] - y, min_val)
            boxes[:,2] = minimum(boxes[:,2] - x, maxw_val)
            boxes[:,3] = minimum(boxes[:,3] - y, maxh_val)
            
        boxes *= strong_aug_info["ratio"]
        
        box_h = boxes[:,3] - boxes[:,1]
        box_w = boxes[:,2] - boxes[:,0]
        ratio = box_w / box_h
        
        not_small_boxes = where(
            (box_h / maxh_val > 0.025) & 
            (box_w / maxw_val > 0.025) & 
            (ratio <= 4) & (ratio >= 0.25)
        )[0]
        boxes = boxes[not_small_boxes, :]
   
        # TODO: check if the boxes are toooooo small in some length
        
        if type(boxes) == np.ndarray:
            boxes = boxes.astype(int)
        else:
            boxes = boxes.long()
        
        return boxes
    
    
    def load_image(self, index):
    
        img = None
        min_crop_ratio = 1                  
        while img is None:

            rand_num = np.random.rand()
            if rand_num <= 0.35:
                img_path = self.load_golden_file(self.files)
                min_crop_ratio = 0.5
            elif rand_num <= 0.60:
                img_path = self.load_m109_file(self.files)
                min_crop_ratio = 0.3
            elif rand_num <= 0.95:
                img_path = self.load_icf_file(self.files)
                min_crop_ratio = 0.6
            else:
                img_path = self.load_2k_set_file(self.files)
                min_crop_ratio = 0.6
    
            if not os.path.isfile(img_path):
                print("[ERROR] The file does not exist:", img_path)
            
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
        assert img is not None
        h, w, c = img.shape
        
        return img, [h, w], min_crop_ratio
    
    def load_2k_set_file(self, paths):
        partition = np.random.choice(["comic", "watercolor", "clipart"])
        twok_path = paths[partition]
        files = os.listdir(twok_path)
        file = files[np.random.randint(len(files))]
        
        if "ipynb" in file:
            return self.load_2k_set_file(paths)
        else:
            return os.path.join(twok_path, file)
    
    
    def load_m109_file(self, paths):
        m109_path = paths["m109"]
        folders = os.listdir(m109_path)
        folder = folders[np.random.randint(len(folders))]
        files = os.listdir(os.path.join(m109_path, folder))
        file = files[np.random.randint(len(files))]
        
        if "ipynb" in file or "ipynb" in folder:
            return self.load_m109_file(paths)
        else:
            return os.path.join(m109_path, folder, file)
    
    
    def load_golden_file(self, paths):
        golden_path = paths["golden"]
        folders = os.listdir(golden_path)
        
        files = None
        while files is None:
            folder = folders[np.random.randint(len(folders))]
            files = os.listdir(os.path.join(golden_path, folder))
            if len(files) == 0:
                print("[WARNING] No file in:", os.path.join(golden_path, folder))
                files = None
        
        file = files[np.random.randint(len(files))]
        
        if "ipynb" in file or "ipynb" in folder:
            return self.load_golden_file(paths)
        else:
            return os.path.join(golden_path, folder, file)


    def load_icf_file(self, paths):
        icf_path = paths["icf"]
        files = os.listdir(icf_path)
        file = files[np.random.randint(len(files))]
        
        if "ipynb" in file:
            return self.load_icf_file(paths)
        else:
            return os.path.join(icf_path, file)