import os
from collections import OrderedDict
from loguru import logger

import cv2
import numpy as np

from ..dataloading import get_yolox_datadir
from .datasets_wrapper import Dataset
from ..data_augment import draw_speech_balloon


class WiderFaceDataset(Dataset):
    """
    Widerface dataset class.
    """

    def __init__(
        self,
        data_dir=None,
        train=True,
        img_size=(416, 416),
        preproc=None,
        cache=False, # no cache is supported currently
        filter_file=None,
        random_style="all",
        **kwargs
    ):
        super().__init__(img_size)
        self.class_ids = [0]
        self.train_mode = train
        self.class_dict = {"face" : self.class_ids[0]}
        self.img_size = img_size
        self.preproc = preproc   
        self.files, self.annotations = self.load_annotations(data_dir, train)
        
        self.set_style(random_style)
        
        self.len_ratio = None
        if filter_file is not None and train:
            
            wf_path = data_dir["wf_train_imgs"] if train else data_dir["wf_test_imgs"]
            
            f = open(filter_file, "r")
            lines = f.readlines()
            if len(lines[-1]) < 2:
                lines = lines[:-1]
            
            if lines[0][-1] == "\n":
                    lines[0] = lines[0][:-1]
            
            self.len_ratio = float(lines[0])
            
            files = []
            for i, line in enumerate(lines[1:]):
                if line[-1] == "\n":
                    line = line[:-1]
                    
                final_file = line
                files.append(final_file)
            
            self.files = files
        
        
    def __len__(self):
        return len(self.files)
    
    
    def set_style(self, random_style):
        if random_style == "best_comb":
            self.random_style = [
                "cartoongan_hosoda", "cartoongan_hayao", 
                "cartoongan_shinkai", "ganilla_KH", "whitebox"
            ]
        else:
            self.random_style = random_style
    
    def pull_item(self, index):
        img, img_info = self.load_image(index)
        res = self.load_anno(index)
        
        if self.len_ratio is not None:
            lens = (res[:,2:4] - res[:,0:2]) / img_info
            selected_indices = np.where(lens > self.len_ratio)[0]
            res = res[selected_indices, :]
        
#         if self.train_mode and np.random.rand() > 0.5:
#             img = draw_speech_balloon(img, add_noise=np.random.rand() > 0.5)
        
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
        
        img_file = self.files[index]
        orig_img_path = None
        if self.train_mode:
            
            if self.random_style == "all":
                different_styles = os.listdir(self.img_root)
                random_style = different_styles[np.random.randint(len(different_styles))]
            elif type(self.random_style) == str:
                random_style = self.random_style
            else:
                random_style = np.random.choice(self.random_style)
            
            img_path = os.path.join(self.img_root, random_style, img_file)
            orig_img_path = os.path.join(self.img_root, "original", img_file)
        else:
            img_path = os.path.join(self.img_root, img_file)

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        assert img is not None
        
        if orig_img_path is not None:
            orig_img = cv2.imread(orig_img_path, cv2.IMREAD_COLOR)
            if img.shape != orig_img.shape:
                h, w, c = orig_img.shape
                img = cv2.resize(img, [w, h]) 
        
        h, w, c = img.shape
        return img, [h, w]
    
    
    def load_annotations(self, wf_paths, train :bool):
        
        # given a path and partition, it loads all the image paths and annots in that partition
        files = []
        boxes = {}
        
        wf_path = wf_paths["wf_train_imgs"] if train else wf_paths["wf_test_imgs"]
        labels_path = wf_paths["wf_train_labels"] if train else wf_paths["wf_test_labels"]
        
        self.img_root = wf_path
        
        f = open(labels_path, 'r')
        lines = f.readlines()
        f.close()

        img_boxes = []; last_path = None
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                if last_path is not None:
                    boxes[last_path] = np.array(img_boxes)
                    img_boxes = []
                path = line[2:]
                # path = os.path.join(wf_path, path)
                
                last_path = path
                files.append(path)
            else:
                line = line.split(' ')
                labels = [float(x) for x in line]
                person_annot = np.zeros(4)
                person_annot[0:4] = labels[0:4]
                person_annot[2:4] += person_annot[0:2]            
                img_boxes.append(person_annot)

        boxes[last_path] = np.array(img_boxes)    
        
        return files, boxes