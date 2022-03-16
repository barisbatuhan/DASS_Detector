import os
from collections import OrderedDict
from loguru import logger

import cv2
import json
import random
import numpy as np

from ..dataloading import get_yolox_datadir
from .datasets_wrapper import Dataset


class ComicFacesDataset(Dataset):
    """
    A compilation dataset (icartoonface, manga109 frames, golden faces) for faces.
    """

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
        self.class_ids = [0]
        self.class_dict = {"face" : self.class_ids[0]}
        self.img_size = img_size
        self.preproc = preproc   
        
        self.data_ratios = {
            'icf': 0.85,
            'm109': 0.1,
            'dcm': 0.05
        }
        self.data_limit = limit_dataset
        
        if not train:
            self.files, self.annotations = self.load_icf_annotations(data_dir, train)
        else:
            
            self.icf_files, self.annotations = self.load_icf_annotations(data_dir, train)
            self.icf_files = np.asarray(self.icf_files)
            
            self.dcm_files, annots = self.load_dcm772_annotations(data_dir)
            self.dcm_files = np.asarray(self.dcm_files)
            self.annotations.update(annots)
            
            self.m109_frames_files, annots = self.load_manga109_frames_annotations(data_dir, train)
            self.m109_frames_files = np.asarray(self.m109_frames_files)
            self.annotations.update(annots)
            
            self.files = []
            self.arrange_files(data_dir) # selects the files  
        
    def arrange_files(self, data_dir=None): 
        
        if self.data_limit is None: # 10240 images
            
            self.files = []
        
            chosen = np.random.choice(len(self.icf_files), 8704, replace=False)
            self.files.extend(self.icf_files[chosen])
            
            chosen = np.random.choice(len(self.m109_frames_files), 1024, replace=False)
            self.files.extend(self.m109_frames_files[chosen])
            
            chosen = np.random.choice(len(self.dcm_files), 512, replace=False)
            self.files.extend(self.dcm_files[chosen])
         
        else:
            
            if self.files is not None and len(self.files) < 1 and data_dir is not None:
                
                with open('datasets/comic_faces_list.json') as json_file:
                    file_list = json.load(json_file)
                
                self.files.extend(
                    [os.path.join(data_dir['icf_train_imgs'], file) for file in file_list['icf'][:int(
                        self.data_limit * self.data_ratios['icf'])]])
                
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
                boxes[filepath] = json_boxes[k]["face"]
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
            box = np.asarray(json_boxes[file]["face"])
            boxes[k] = box[:,:4]
            
        return files, boxes
        
    
    def load_golden_faces_annotations(self, golden_paths):
        files = []
        boxes = {}
        
        img_path = golden_paths["golden_faces_imgs"]
        labels_path = golden_paths["golden_faces_labels"]
        
        f = open(labels_path,'r')
        lines = f.readlines()
        f.close()
        
        img_boxes = []; last_path = None
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                if last_path is not None:
                    boxes[last_path] = np.array(img_boxes)
                    img_boxes = []
                path = line[3:]

                last_path = path
                files.append(path)
            else:
                line = line.split(' ')
                labels = [float(x) for x in line]
                person_annot = np.zeros(4)
                person_annot[0:4] = labels[0:4]
                # person_annot[2:4] += person_annot[0:2]            
                img_boxes.append(person_annot)

        boxes[last_path] = np.array(img_boxes)
        return files, boxes
    
    
    def load_icf_annotations(self, icf_paths, train :bool):
        # gven a path and partition, it loads all the image paths and annots in that partition
        files = []
        boxes = {}
        
        icf_path = icf_paths["icf_train_imgs"] if train else icf_paths["icf_test_imgs"]
        labels_path = icf_paths["icf_train_labels"] if train else icf_paths["icf_test_labels"]
        
        if labels_path is not None:
            
            f = open(labels_path,'r')
            lines = f.readlines()
            f.close()
    
            for line in lines:
                line = line.rstrip().split(',')
                labels = [float(x) for x in line[1:5]]
                person_annot = np.zeros(4)
                person_annot[0:4] = labels[0:4]
                if icf_path + line[0] not in boxes:
                    boxes[icf_path + line[0]] = []
                boxes[icf_path + line[0]].append(person_annot)
            
            for k in boxes.keys():
                boxes[k] = np.array(boxes[k])
            
            files = [*boxes.keys()]
        
        else:
            files = [os.path.join(icf_path, file) for file in os.listdir(icf_path)]
            for file in files:
                boxes[file] = None

        return files, boxes
        