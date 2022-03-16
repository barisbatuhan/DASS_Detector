import os
from collections import OrderedDict
from loguru import logger

import cv2
import json
import random
import numpy as np

from ..dataloading import get_yolox_datadir
from .datasets_wrapper import Dataset


class ComicBodiesDataset(Dataset):
    """
    A compilation dataset (manga109 frames, comic2k, watercolor2k, clipart1k) for bodies.
    """

    def __init__(
        self,
        data_dir=None,
        train=True,
        img_size=(416, 416),
        preproc=None,
        cache=False, # no cache is supported currently
        filter=None,
        include_animals=True,
        include_back_chars=True,
        limit_dataset=None
    ):
        super().__init__(img_size)
        self.class_ids = [0]
        self.class_dict = {"face" : self.class_ids[0]}
        self.img_size = img_size
        self.preproc = preproc  
        
        self.data_ratios = {
            'comic': 0.04,
            'watercolor': 0.04,
            'clipart': 0.02,
            'm109': 0.8,
            'dcm': 0.1
        }
        self.data_limit = limit_dataset
        
        self.include_animals = include_animals
        self.include_back_chars = include_back_chars
        
        if not train: # equal sized data for each dataset
            self.files, self.annotations = self.load_comic2k_annotations(data_dir, train, filter=filter)
            self.m109_frames_files, annots = self.load_manga109_frames_annotations(data_dir, train)
            self.m109_frames_files = np.asarray(self.m109_frames_files)
            
            chosen = np.random.choice(len(self.m109_frames_files), len(self.files), replace=False)
            self.annotations.update(annots)
            self.files.extend(self.m109_frames_files[chosen])
            
        else:
            
            self.comic2k_files, self.annotations = self.load_comic2k_annotations(data_dir, train, filter=filter)
            self.comic2k_files = np.asarray(self.comic2k_files)

            self.m109_frames_files, annots = self.load_manga109_frames_annotations(data_dir, train)
            self.m109_frames_files = np.asarray(self.m109_frames_files)
            self.annotations.update(annots)
            
            self.dcm_files, annots = self.load_dcm772_annotations(data_dir)
            self.dcm_files = np.asarray(self.dcm_files)
            self.annotations.update(annots)
            
            self.files = []
            self.arrange_files(data_dir) # selects the files 

    
    def arrange_files(self, data_dir=None): 
        
        if self.data_limit is None: # 10240 images
            
            self.files = []
            
            chosen = np.random.choice(len(self.comic2k_files), 1024, replace=False) # has 1942 images in total
            self.files.extend(self.comic2k_files[chosen])
            
            chosen = np.random.choice(len(self.m109_frames_files), 8192, replace=False)
            self.files.extend(self.m109_frames_files[chosen])
            
            chosen = np.random.choice(len(self.dcm_files), 1024, replace=False)     # has 2335 images in total
            self.files.extend(self.dcm_files[chosen])
         
        else:
            
            if self.files is not None and len(self.files) < 1 and data_dir is not None:
                
                with open('datasets/comic_bodies_list.json') as json_file:
                    file_list = json.load(json_file)
            
                self.files.extend(
                    [os.path.join(data_dir['comic'], file) for file in file_list['comic'][:int(
                        self.data_limit * self.data_ratios['comic'])]])
                
                self.files.extend(
                    [os.path.join(data_dir['watercolor'], file) for file in file_list['watercolor'][:int(
                        self.data_limit * self.data_ratios['watercolor'])]])
                
                self.files.extend(
                    [os.path.join(data_dir['clipart'], file) for file in file_list['clipart'][:int(
                        self.data_limit * self.data_ratios['clipart'])]])
                
                self.files.extend(
                    [os.path.join(data_dir['m109_frames_imgs'], file) for file in file_list['m109'][:int(
                        self.data_limit * self.data_ratios['m109'])]])
                
                self.files.extend(
                    [os.path.join(data_dir['dcm772_frames_imgs'], file) for file in file_list['dcm'][:int(
                        self.data_limit * self.data_ratios['dcm'])]])
                
        random.shuffle(self.files)
        
        
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

        for idx, face in enumerate(annots):
            anno[idx,:4] = face
            anno[idx,4] = self.class_dict["face"]
        
        return anno

    def load_image(self, index):
        img_path = self.files[index]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        assert img is not None
        h, w, c = img.shape
        return img, [h, w]
    
    
    def load_manga109_frames_annotations(self, m109_frames_paths, train :bool):
        img_path = m109_frames_paths["m109_frames_imgs"]
        labels_path = m109_frames_paths["m109_frames_labels"]
        
        test_books = ["UltraEleven", "UnbalanceTokyo", "WarewareHaOniDearu", "YamatoNoHane", "YasasiiAkuma", 
                      "YouchienBoueigumi", "YoumaKourin", "YukiNoFuruMachi", "YumeNoKayoiji", "YumeiroCooking"]
        
        def from_testset(file, test_books):
            for t in test_books:
                if t in file:
                    return True
            return False
        
        with open(labels_path, "r") as f:
            json_boxes = json.load(f)
         
        files = []
        boxes = {}
        for k in json_boxes.keys():
            if (train and not from_testset(k, test_books)) or (not train and from_testset(k, test_books)):
                filepath = os.path.join(img_path, k)
                boxes[filepath] = json_boxes[k]["body"]
                files.append(filepath)
        
        return files, boxes
    
    
    def load_dcm772_annotations(self, dcm_paths):
        files = []
        boxes = {}
        
        img_path = dcm_paths["dcm772_frames_imgs"]
        labels_path = dcm_paths["dcm772_frames_labels"]
        partition_path = dcm_paths["dcm772_frames_partition"]
        
        with open(labels_path, "r") as f:
            json_boxes = json.load(f)
        
        with open(partition_path, "r") as f:
            files_list = f.readlines()
        
        for file in files_list:
            if len(file) <= 2:
                continue
            elif file[-1] == "\n":
                file = file[:-1]
            
            k = os.path.join(img_path, file)
            files.append(k)
            box = np.asarray(json_boxes[file]["body"])
            classes = box[:,-1]
            
            filter_class = [1]
            if self.include_animals:
                filter_class.append(5)
            if self.include_back_chars:
                filter_class.append(6)
            
            selected = []
            for i in range(classes.shape[0]):
                if classes[i] in filter_class:
                    selected.append(i)
            
            selected = np.asarray(selected)
            
            boxes[k] = box[selected,:4]
            
        return files, boxes
    
    
    def load_comic2k_annotations(self, comic2k_paths, train :bool, filter):
        # gven a path and partition, it loads all the image paths and annots in that partition
        files = []
        boxes = {}
        
        annot_final = "train.txt" if train else "test.txt"
        
        # Comic Part Reading
        if filter is None or filter == "comic":
            f = open(os.path.join(comic2k_paths["comic"], "ImageSets/Main", "annotated_" +  annot_final), "r")
            comic2k_files = f.readlines()
            f.close()
            boxes["comic"] = self.__read_xmls__(os.path.join(comic2k_paths["comic"], "Annotations"), comic2k_files)
            
            prev_keys = list(boxes["comic"].keys())
            for file in prev_keys:
                k = os.path.join(comic2k_paths["comic"], "JPEGImages", file + ".jpg")
                boxes[k] = boxes["comic"].pop(file)
                files.append(k)
            boxes.pop("comic")
        
        # Watercolor Part Reading
        if filter is None or filter == "watercolor":
            f = open(os.path.join(comic2k_paths["watercolor"], "ImageSets/Main", "annotated_" +  annot_final), "r")
            comic2k_files = f.readlines()
            f.close()
            boxes["watercolor"] = self.__read_xmls__(os.path.join(comic2k_paths["watercolor"], "Annotations"), comic2k_files)
            
            prev_keys = list(boxes["watercolor"].keys())
            for file in prev_keys:
                k = os.path.join(comic2k_paths["watercolor"], "JPEGImages", file + ".jpg")
                boxes[k] = boxes["watercolor"].pop(file)
                files.append(k)
            boxes.pop("watercolor")
        
        
        # Clipart Part Reading
        if filter is None or filter == "clipart":
            f = open(os.path.join(comic2k_paths["clipart"], "ImageSets/Main", annot_final), "r")
            comic2k_files = f.readlines()
            f.close()
            boxes["clipart"] = self.__read_xmls__(os.path.join(comic2k_paths["clipart"], "Annotations"), comic2k_files)
            
            prev_keys = list(boxes["clipart"].keys())
            for file in prev_keys:
                k = os.path.join(comic2k_paths["clipart"], "JPEGImages", file + ".jpg")
                boxes[k] = boxes["clipart"].pop(file)
                files.append(k)
            boxes.pop("clipart")

        return files, boxes
    
    
    def __read_xmls__(self, annot_path, files):
        
        boxes = {}
        
        for annot in files:
            anno = annot[:-1]
            boxes[anno] = []
            past_name = None
            new_box = [0, 0, 0, 0]
            
            f = open(os.path.join(annot_path, anno + ".xml"), "r")
            lines = f.readlines()
            f.close()
            for line in lines:
                start = line.find(">") + 1
                end = line.find("</")
                content = line[start:end]
                if "<name>" in line:
                    if past_name == "person":
                        if (new_box[3] - new_box[1]) * (new_box[2] - new_box[0]) > 16:
                            # eliminate really small people annotations
                            boxes[anno].append(new_box)
                    
                    new_box = [0, 0, 0, 0]
                    past_name = content
                
                elif past_name == "person":
                    if "xmin" in line:
                        new_box[0] = int(content)
                    elif "ymin" in line:
                        new_box[1] = int(content)
                    elif "xmax" in line:
                        new_box[2] = int(content)
                    elif "ymax" in line:
                        new_box[3] = int(content)
                
                
            if new_box[-1] != 0 and new_box not in boxes[anno] and (new_box[3] - new_box[1]) * (new_box[2] - new_box[0]) > 16:
                boxes[anno].append(new_box)
        
        all_files = list(boxes.keys())
        for anno in all_files:
             if len(boxes[anno]) == 0:
                boxes.pop(anno)
        
        return boxes