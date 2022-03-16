import importlib
import os
import sys

from .comic_exp import ComicExp
from .styled_exp import AllStyledExp, AllNoBubbleStyledExp, AllOnlyPersonStyledExp, NoneStyledExp, BestStyledExp, CustomStyledExp
from .single_dataset_exp import SingleDatasetExp
from .unsupervised_exp import (UnsupervisedExp, 
                               Match250ConstUnsupervisedExp, Match500ConstUnsupervisedExp, Match1000ConstUnsupervisedExp, 
                               Match2000ConstUnsupervisedExp, Match5000ConstUnsupervisedExp, NoMatchConstUnsupervisedExp,
                               FocalConstUnsupervisedExp, FocalNoMatchConstUnsupervisedExp, MovingUnsupervisedExp,
                               EMA9990UnsupervisedExp, EMA9992UnsupervisedExp, EMA9996UnsupervisedExp, 
                               EMA9998UnsupervisedExp, EMA9999UnsupervisedExp,
                               NoRegUnsupervisedExp, Reg1UnsupervisedExp, Reg4UnsupervisedExp, Reg10UnsupervisedExp,
                               StuPos50Neg50UnsupervisedExp, StuPos70Neg30UnsupervisedExp, StuPos30Neg70UnsupervisedExp, 
                               StuPos05Neg95UnsupervisedExp, StuPos00Neg100UnsupervisedExp,
                               TeacConf50UnsupervisedExp, TeacConf75UnsupervisedExp, TeacConf35UnsupervisedExp, TeacConf90UnsupervisedExp,
                              )


def init_exp_by_process_name(process_name :str):
    
    available_processes = {
        "comic": ComicExp,
        "styled": AllStyledExp,
        "allstyled": AllStyledExp,
        "beststyled": BestStyledExp,
        "nonestyled": NoneStyledExp,
        "nobubblestyled": AllNoBubbleStyledExp,
        "onlypersonstyled": AllOnlyPersonStyledExp,
        "unsupervised": UnsupervisedExp,
        "icartoonface": SingleDatasetExp,
        "manga109": SingleDatasetExp,
        "dcm772": SingleDatasetExp,
        "comic2k": SingleDatasetExp,
        "cartoongan_hayao": CustomStyledExp, 
        "cartoongan_shinkai": CustomStyledExp, 
        "cyclegan_vangogh": CustomStyledExp, 
        "ganilla_miyazaki": CustomStyledExp, 
        "cartoongan_hosoda": CustomStyledExp, 
        "cyclegan_cezanne": CustomStyledExp, 
        "ganilla_AS": CustomStyledExp, 
        "cartoongan_paprika": CustomStyledExp, 
        "cyclegan_monet": CustomStyledExp, 
        "ganilla_KH": CustomStyledExp, 
        "whitebox": CustomStyledExp,
        "uns_const250": Match250ConstUnsupervisedExp, 
        "uns_const500": Match500ConstUnsupervisedExp, 
        "uns_const1000": Match1000ConstUnsupervisedExp, 
        "uns_const2000": Match2000ConstUnsupervisedExp, 
        "uns_const5000": Match5000ConstUnsupervisedExp, 
        "uns_constnomatch": NoMatchConstUnsupervisedExp,
        "uns_const_focal": FocalConstUnsupervisedExp, 
        "uns_const_focal_nomatch": FocalNoMatchConstUnsupervisedExp, 
        "uns_moving": MovingUnsupervisedExp,
        "uns_ema9990": EMA9990UnsupervisedExp, 
        "uns_ema9992": EMA9992UnsupervisedExp, 
        "uns_ema9996": EMA9996UnsupervisedExp, 
        "uns_ema9998": EMA9998UnsupervisedExp, 
        "uns_ema9999": EMA9999UnsupervisedExp,
        "uns_noreg": NoRegUnsupervisedExp, 
        "uns_reg1": Reg1UnsupervisedExp, 
        "uns_reg4": Reg4UnsupervisedExp, 
        "uns_reg10": Reg10UnsupervisedExp,
        "uns_stu50_50": StuPos50Neg50UnsupervisedExp,
        "uns_stu70_30": StuPos70Neg30UnsupervisedExp, 
        "uns_stu30_70": StuPos30Neg70UnsupervisedExp,                           
        "uns_stu05_95": StuPos05Neg95UnsupervisedExp, 
        "uns_stu00_100": StuPos00Neg100UnsupervisedExp,
        "uns_teac_50": TeacConf50UnsupervisedExp, 
        "uns_teac_75": TeacConf75UnsupervisedExp, 
        "uns_teac_35": TeacConf35UnsupervisedExp, 
        "uns_teac_90": TeacConf90UnsupervisedExp
    }
    assert process_name in available_processes.keys()
    return available_processes[process_name]