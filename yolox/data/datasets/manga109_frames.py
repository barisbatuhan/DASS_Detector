import os
from collections import OrderedDict
from loguru import logger

import cv2
import json
import random
import numpy as np

from ..dataloading import get_yolox_datadir
from .datasets_wrapper import Dataset


class Manga109FramesDataset(Dataset):

    def __init__(
        self,
        data_dir=None,
        train=True,
        img_size=(416, 416),
        preproc=None,
        cache=False, # no cache is supported currently
        limit_dataset=None
    ):
        super().__init__(img_size)
        self.class_ids = [0, 1]
        self.class_dict = {"face" : self.class_ids[0], "body": self.class_ids[1]}
        self.img_size = img_size
        self.preproc = preproc   
        self.data_limit = limit_dataset
            
        self.files, self.annotations = self.load_annotations(data_dir, train)
        self.files = np.asarray(self.files)

        if limit_dataset is not None:
            self.files = self.files[np.random.choice(len(self.files), 
                                                     min(limit_dataset, len(self.files)),
                                                     replace=False)]
  
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
        return annots

    def load_image(self, index):
        img_path = self.files[index]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        assert img is not None
        h, w, c = img.shape
        return img, [h, w]
    
    def load_annotations(self, m109_frames_paths, train :bool):
        img_path = m109_frames_paths["m109_frames_imgs"]
        labels_path = m109_frames_paths["m109_frames_labels"]
        
        test_books = ["UltraEleven", "UnbalanceTokyo", "WarewareHaOniDearu", 
                      "YamatoNoHane", "YasasiiAkuma", "YouchienBoueigumi", "YoumaKourin", 
                      "YukiNoFuruMachi", "YumeNoKayoiji", "YumeiroCooking"]
        
        def from_testset(file, test_books):
            for t in test_books:
                if t in file:
                    return True
            return False
        
        with open(labels_path, "r") as f:
            json_boxes = json.load(f)
         
        files, boxes = [], {}
        
        for k in json_boxes.keys():
            if (train and not from_testset(k, test_books)) or (not train and from_testset(k, test_books)):
                filepath = os.path.join(img_path, k)
                faces, bodies = np.asarray(json_boxes[k]["face"]), np.asarray(json_boxes[k]["body"])

                if faces.shape[0] > 0:
                    fcls_ids = np.zeros((faces.shape[0], 1))
                    fcls_ids.fill(self.class_dict["face"])
                    boxes[filepath] = np.concatenate([faces[:,:4], fcls_ids], axis=1)
                else:
                    boxes[filepath] = np.zeros((0, 5))
            
                if bodies.shape[0] > 0:
                    bcls_ids = np.zeros((bodies.shape[0], 1))
                    bcls_ids.fill(self.class_dict["body"])
                    boxes[filepath] = np.concatenate([
                        boxes[filepath],
                        np.concatenate([bodies[:,:4], bcls_ids], axis=1)
                    ], axis=0)
                    
                files.append(filepath)
        
        return files, boxes