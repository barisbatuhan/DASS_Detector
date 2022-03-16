#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

# from .coco import COCODataset
from .coco_persons import COCOPersonsDataset
from .coco_classes import COCO_CLASSES
from .datasets_wrapper import ConcatDataset, Dataset, MixConcatDataset
from .mosaicdetection import MosaicDetection
from .manga109 import Manga109Dataset
from .icartoonface import ICartoonFaceDataset
from .widerface import WiderFaceDataset
from .ebdtheque_frames import EBDthequeFramesDataset
from .manga109_frames import Manga109FramesDataset
from .dcm772_frames import DCM772FramesDataset
from .ebdtheque import EBDthequeDataset
from .comic2k import Comic2kDataset
from .dcm772 import DCM772Dataset

from .comic_faces import ComicFacesDataset
from .comic_bodies import ComicBodiesDataset

from .raw_unsupervised_comics import RawUnsupervisedComicsDataset
