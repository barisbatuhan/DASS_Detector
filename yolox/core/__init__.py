from .comic_trainer import ComicTrainer
from .unsupervised_trainer import UnsupervisedTrainer
from .single_dataset_trainer import SingleDatasetTrainer

__all__ = ["init_trainer_by_process_name"]

def init_trainer_by_process_name(process_name :str):
    
    available_processes = {
        "comic": ComicTrainer,
        "styled": ComicTrainer,
        "allstyled": ComicTrainer,
        "beststyled": ComicTrainer,
        "nonestyled": ComicTrainer,
        "nobubblestyled": ComicTrainer,
        "onlypersonstyled": ComicTrainer,
        "unsupervised": UnsupervisedTrainer,
        "icartoonface": SingleDatasetTrainer,
        "manga109": SingleDatasetTrainer,
        "dcm772": SingleDatasetTrainer,
        "comic2k": SingleDatasetTrainer,
        "cartoongan_hayao": ComicTrainer, 
        "cartoongan_shinkai": ComicTrainer, 
        "cyclegan_vangogh": ComicTrainer, 
        "ganilla_miyazaki": ComicTrainer, 
        "cartoongan_hosoda": ComicTrainer, 
        "cyclegan_cezanne": ComicTrainer, 
        "ganilla_AS": ComicTrainer, 
        "cartoongan_paprika": ComicTrainer, 
        "cyclegan_monet": ComicTrainer, 
        "ganilla_KH": ComicTrainer, 
        "whitebox": ComicTrainer,
        "uns_const250": UnsupervisedTrainer, 
        "uns_const500": UnsupervisedTrainer, 
        "uns_const1000": UnsupervisedTrainer, 
        "uns_const2000": UnsupervisedTrainer, 
        "uns_const5000": UnsupervisedTrainer, 
        "uns_constnomatch": UnsupervisedTrainer,
        "uns_const_focal": UnsupervisedTrainer, 
        "uns_const_focal_nomatch": UnsupervisedTrainer, 
        "uns_moving": UnsupervisedTrainer,
        "uns_ema9990": UnsupervisedTrainer, 
        "uns_ema9992": UnsupervisedTrainer, 
        "uns_ema9996": UnsupervisedTrainer, 
        "uns_ema9998": UnsupervisedTrainer, 
        "uns_ema9999": UnsupervisedTrainer,
        "uns_noreg": UnsupervisedTrainer, 
        "uns_reg1": UnsupervisedTrainer, 
        "uns_reg4": UnsupervisedTrainer, 
        "uns_reg10": UnsupervisedTrainer,
        "uns_stu50_50": UnsupervisedTrainer,
        "uns_stu70_30": UnsupervisedTrainer, 
        "uns_stu30_70": UnsupervisedTrainer,                           
        "uns_stu05_95": UnsupervisedTrainer, 
        "uns_stu00_100": UnsupervisedTrainer,
        "uns_teac_50": UnsupervisedTrainer, 
        "uns_teac_75": UnsupervisedTrainer, 
        "uns_teac_35": UnsupervisedTrainer, 
        "uns_teac_90": UnsupervisedTrainer
    }
    
    assert process_name in available_processes.keys()
    return available_processes[process_name]