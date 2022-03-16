import os
import json
import copy
from collections import OrderedDict

import numpy as np
from tqdm import tqdm
from PIL import Image

def get_faces_and_bodies_in_frame(frame, faces, bodies, margin=0.5):
    fx1, fy1, fx2, fy2, _ = frame
    area = (fx2 - fx1) * (fy2 - fy1)
    
    if area < 1:
        return None, None
    
    in_faces, in_bodies = [], []
    
    for face in faces:
        x1, y1, x2, y2, cls = face
        
        if (x2 - x1) * (y2 - y1) <= 0:
            continue
        
        intersection = max(0, min(fx2, x2) - max(fx1, x1)) * max(0, min(fy2, y2) - max(fy1, y1))
        min_area = min(area, (x2 - x1) * (y2 - y1))
        
        if intersection / min_area > margin:
            in_faces.append([max(0, x1 - fx1), 
                             max(0, y1 - fy1), 
                             min(fx2 - fx1, x2 - fx1), 
                             min(fy2 - fy1, y2 - fy1),
                             cls
                            ])
    
    for body in bodies:
        x1, y1, x2, y2, cls = body
        
        if (x2 - x1) * (y2 - y1) <= 0:
            continue
        
        intersection = max(0, min(fx2, x2) - max(fx1, x1)) * max(0, min(fy2, y2) - max(fy1, y1))
        min_area = min(area, (x2 - x1) * (y2 - y1))
        
        if intersection / min_area > margin:
            in_bodies.append([max(0, x1 - fx1),
                              max(0, y1 - fy1), 
                              min(fx2 - fx1, x2 - fx1), 
                              min(fy2 - fy1, y2 - fy1),
                              cls
                             ])
    
    in_faces = None if in_faces == [] else in_faces
    in_bodies = None if in_bodies == [] else in_bodies
    
    return in_faces, in_bodies


def load_annotations(dcm_path, group):
        # given a path and partition, it loads all the image paths and annots in that partition
        
        boxes = {}
        
        cls_maps = {1: "body", 5: "body", 6: "body", 7:"face", 8:"frame"}
        
        files = []
        img_path    = os.path.join(dcm_path, "images")
        labels_path = os.path.join(dcm_path, "groundtruth")
        
        for file in group:
            # changes in file
            if len(file) < 2:
                break
            elif file[-1] == "\n":
                file = file[:-1]
            elif file[-4:].lower() in [".jpg", ".txt", ".png"]:
                file = file[:-4]
            
            annot_file = os.path.join(labels_path, file + ".txt")
            boxes[file] = {"frame":[], "face":[], "body":[]}
            
            with open(annot_file, "r") as f:
                lines = f.readlines()
            
            for line in lines:
                if len(line) < 2:
                    continue
                elif line[-1] == "\n":
                    line = line[:-1]
                
                cls, x1, y1, x2, y2 = line.split(" ")
                cls, x1, y1, x2, y2 = int(cls), int(x1), int(y1), int(x2), int(y2)
                if cls in cls_maps.keys():
                    boxes[file][cls_maps[cls]].append([x1, y1, x2, y2, cls])
                    
        return boxes




dcm_path = "/userfiles/comics_grp/dcm772/dcm-dataset_from_rigaud/"
save_path = "./"
frame_ratio = 3

with open(os.path.join(dcm_path, "train.txt"), "r") as f:
    train_files = f.readlines()

with open(os.path.join(dcm_path, "val.txt"), "r") as f:
    val_files = f.readlines()

with open(os.path.join(dcm_path, "test.txt"), "r") as f:
    test_files = f.readlines()


annot = {}
for i, group in enumerate([train_files, val_files, test_files]):
    
    if i == 0:
        file_list_save = os.path.join(save_path, "annots", "train.txt")
    elif i == 1:
        file_list_save = os.path.join(save_path, "annots", "val.txt")
    elif i == 2:
        file_list_save = os.path.join(save_path, "annots", "test.txt")
    
    boxes = load_annotations(dcm_path, group)
    
    for page in tqdm(boxes.keys()):   
        
        series, filename = page.split("/")
        
        save_folder = os.path.join(save_path, "imgs", series)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
            
        img = Image.open(os.path.join(dcm_path, "images", page + ".jpg")).convert("RGB")
        
        for fidx, frame in enumerate(boxes[page]["frame"]):
                faces, bodies = get_faces_and_bodies_in_frame(copy.deepcopy(frame), 
                                                              copy.deepcopy(boxes[page]["face"]), 
                                                              copy.deepcopy(boxes[page]["body"]))
                
                if faces is not None and bodies is not None and len(faces) > 0 and len(bodies) > 0:            
                    cpy_img = copy.deepcopy(img)
                    cpy_img = cpy_img.crop(frame[:4])
                    new_img_path = page + "_" + str(fidx) + ".jpg"
                    cpy_img.save(os.path.join(save_path, "imgs", new_img_path))
                    
                    with open(file_list_save, "a") as f_part:
                        f_part.write(new_img_path + "\n")
                    
                    annot[new_img_path] = {
                        "face": copy.deepcopy(faces),
                        "body": copy.deepcopy(bodies)
                    }
    
if annot:
    with open(os.path.join(save_path, "annots", "annotations.json"), 'w') as fp:
        json.dump(annot, fp)