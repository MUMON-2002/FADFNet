import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
from glob import glob
import torch
from .custom_dataset import Custom_Dataset
from sklearn.model_selection import train_test_split

DATASET_INFO = {
    "mayo16": {
        "dir": "/data20tb1/qfj/Datasets/CT/mayo16/data_file/",
        "input_label": "*_input.npy", 
        "target_label": "*_target.npy",
        "test_patients": ['L506']
    },
    "mayo20": {
        "dir": "/data20tb1/qfj/Datasets/CT/mayo20/data_file/",
        "input_label": "*_input.npy",
        "target_label": "*_target.npy",
        "test_patients": ['L212', 'L248']
    },
    "bern_1_50": {
        "dir": "/data20tb1/qfj/Datasets/PET/bern_1_50/data_file/",
        "input_label": "*_input.npy", 
        "target_label": "*_target.npy",
        "test_patients": ['P188', 'P189', 'P190', 'P191', 'P192', 'P193', 'P194', 'P195', 'P196', 'P197', 'P198', 'P199', 'P200', 'P201', 'P202']
    },
    "bern_1_100": {
        "dir": "/data20tb1/qfj/Datasets/PET/bern_1_100/data_file/",
        "input_label": "*_input.npy", 
        "target_label": "*_target.npy",
        "test_patients": ['E188', 'E189', 'E190', 'E191', 'E192', 'E193', 'E194', 'E195', 'E196', 'E197', 'E198', 'E199', 'E200', 'E201', 'E202']
    },
    "ui_1_50": {
        "dir": "/data20tb1/qfj/Datasets/PET/ui_1_50/data_file/",
        "input_label": "*_input.npy", 
        "target_label": "*_target.npy",
        "test_patients": ['P295', 'P298', 'P301', 'P302', 'P303', 'P307', 'P308', 'P311', 'P312', 'P313', 'P314', 'P317', 'P318']
    },
    "ui_1_100": {
        "dir": "/data20tb1/qfj/Datasets/PET/ui_1_100/data_file/",
        "input_label": "*_input.npy", 
        "target_label": "*_target.npy",
        "test_patients": ['E295', 'E298', 'E301', 'E302', 'E303', 'E307', 'E308', 'E311', 'E312', 'E313', 'E314', 'E317', 'E318']
    },
    "local_site1": {
        "dir": "/data20tb1/qfj/Datasets/CT/local_cite1/data_file/",
        "input_label": "*_input.npy", 
        "target_label": "*_target.npy",
        "test_patients": ["L00", "L01"]
    },
    "local_site2": {
        "dir": "/data20tb1/qfj/Datasets/CT/local_cite2/data_file/",
        "input_label": "*_input.npy", 
        "target_label": "*_target.npy",
        "test_patients": ["L00", "L01"]
    },
}


class LoaderHook:
    
    def __init__(self, 
                 mode='train', 
                 use_dataset="mayo16",
                 num_workers=16,
                 prefetch_factor=4,
                 switch_epochs=(), 
                 switch_bs=(), 
                 switch_ps=(), 
                 switch_pn=(),
                 train_ratio=0.9,
                 ):
    
        self.mode = mode
        assert mode in ['train', 'test'] 
        assert use_dataset in DATASET_INFO
        
        self.use_dataset = use_dataset
        self.train_ratio = train_ratio
        self.random_seed = 42
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

        self.switch_epochs = switch_epochs
        self.switch_bs = switch_bs
        self.switch_ps = switch_ps
        self.switch_pn = switch_pn 

        self.dataset_info = DATASET_INFO[use_dataset]
        self.test_patients = self.dataset_info["test_patients"]

        assert len(switch_epochs) == len(switch_bs) == len(switch_ps) == len(switch_pn)

        self.is_init = False
        self.batch_size = None
        self.train_dataset = None
        self.val_dataset = None
        
        self.train_input_list = None
        self.train_target_list = None
        self.val_input_list = None
        self.val_target_list = None

    def _split_train_val_data(self):

        if self.train_input_list is not None:
            return 
            
        input_path = sorted(glob(os.path.join(
            self.dataset_info["dir"], 
            self.dataset_info["input_label"]
        )))
        target_path = sorted(glob(os.path.join(
            self.dataset_info["dir"], 
            self.dataset_info["target_label"]
        )))

        all_input_list = [f for f in input_path if not any(p in f for p in self.test_patients)]
        all_target_list = [f for f in target_path if not any(p in f for p in self.test_patients)]

        self.train_input_list, self.val_input_list, self.train_target_list, self.val_target_list = train_test_split(
            all_input_list, all_target_list, 
            train_size=self.train_ratio, 
            random_state=self.random_seed,
            shuffle=True
        )

    def get_dataset(self, patch_n, patch_size):
        
        if self.mode == 'train':
            self._split_train_val_data()
            
            train_dataset = Custom_Dataset(
                input_list=self.train_input_list, 
                target_list=self.train_target_list, 
                patch_n=patch_n, 
                patch_size=patch_size
            )
            
            val_dataset = Custom_Dataset(
                input_list=self.val_input_list, 
                target_list=self.val_target_list, 
                patch_n=patch_n, 
                patch_size=patch_size
            )

            return train_dataset, val_dataset

        elif self.mode == 'test': 
            input_path = sorted(glob(os.path.join(
                self.dataset_info["dir"], 
                self.dataset_info["input_label"]
            )))
            target_path = sorted(glob(os.path.join(
                self.dataset_info["dir"], 
                self.dataset_info["target_label"]
            )))
            
            test_input_list = [f for f in input_path if any(p in f for p in self.test_patients)]
            test_target_list = [f for f in target_path if any(p in f for p in self.test_patients)]
            
            val_dataset = Custom_Dataset(
                input_list=test_input_list, 
                target_list=test_target_list, 
                patch_n=1, 
                patch_size=512
            )
            
            return None, val_dataset

    def get_data_loader(self, epoch=0):
        
        if not self.is_init or epoch in self.switch_epochs:
            
            ind = self.switch_epochs.index(epoch) if epoch in self.switch_epochs else 0
            self.batch_size = self.switch_bs[ind]
            
            self.train_dataset, self.val_dataset = self.get_dataset(
                patch_n=self.switch_pn[ind], 
                patch_size=self.switch_ps[ind]
            )
            
            log_msg = f"[Dataloader Update] {'Init' if not self.is_init else 'New'} dataloader at epoch {epoch}, batch_size={self.switch_bs[ind]}, patch_size={self.switch_ps[ind]}, patch_n={self.switch_pn[ind]}"
            print(log_msg)
            print("=" * 50)

        train_loader = None
        val_loader = None
        
        if self.mode == 'train':

            train_loader = DataLoader(
                self.train_dataset, 
                batch_size=self.batch_size, 
                shuffle=True, 
                num_workers=self.num_workers, 
                prefetch_factor=self.prefetch_factor, 
                pin_memory=True
            )
            
            val_loader = DataLoader(
                self.val_dataset, 
                batch_size=self.batch_size, 
                shuffle=False, 
                num_workers=self.num_workers, 
                prefetch_factor=self.prefetch_factor, 
                pin_memory=True
            )
            
        else:

            train_loader = None
            val_loader = DataLoader(
                self.val_dataset, 
                batch_size=1, 
                shuffle=False, 
                num_workers=self.num_workers, 
                prefetch_factor=self.prefetch_factor, 
                pin_memory=True
            )

        if not self.is_init:
            if self.mode == 'train':
                print(f"Training Dataset ({self.use_dataset}): {len(self.train_dataset)} samples")
                print(f"Validation Dataset ({self.use_dataset}): {len(self.val_dataset)} samples")
            else:
                print(f"Testing Dataset ({self.use_dataset}): {len(self.val_dataset)} samples")
            print("=" * 50)

        self.is_init = True
        return train_loader, val_loader
