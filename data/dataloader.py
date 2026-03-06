import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from glob import glob
from .dataset_info import DATASET_INFO
from .dataset_custom import Custom_Dataset

class Loader_Hook:
    def __init__(self, 
                 mode='train', 
                 num_workers=16, 
                 prefetch_factor=4, 
                 batch_size=16, 
                 use_dataset=None, 
                 patch_size=None, 
                 patch_num=1,
                 context=False,
                 set_val=True 
                ):
        self.mode = mode
        assert mode in ['train', 'test'] 
        assert use_dataset in DATASET_INFO
        
        self.use_dataset = use_dataset
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.patch_num = patch_num
        self.context = context
        self.set_val = set_val

        self.dataset_info = DATASET_INFO[use_dataset]
        self.test_patients = self.dataset_info.get("test_patients", [])
        self.val_patients = self.dataset_info.get("val_patients", [])
        self.data_type = self.dataset_info.get("data_type", "CT")
        self.normalize = (self.data_type == "CT")

        self.train_inputs, self.train_targets = [], []
        self.val_inputs, self.val_targets = [], []
        self.test_inputs, self.test_targets = [], []

        self._distribute_data()

    def _distribute_data(self):
        input_path = sorted(glob(os.path.join(self.dataset_info["dir"], self.dataset_info["input_label"])))
        target_path = sorted(glob(os.path.join(self.dataset_info["dir"], self.dataset_info["target_label"])))
        assert len(input_path) == len(target_path), "Input and target file nums do not match!"

        patient_data = {}
        for inp, tar in zip(input_path, target_path):
            pid = os.path.basename(inp).split('_')[0] 
            if pid not in patient_data:
                patient_data[pid] = {'inputs': [], 'targets': []}
            patient_data[pid]['inputs'].append(inp)
            patient_data[pid]['targets'].append(tar)

        all_pids = list(patient_data.keys())
        
        test_pids = self.test_patients
        if self.set_val:
            val_pids = self.val_patients
            train_pids = [p for p in all_pids if p not in test_pids and p not in val_pids]
        else:
            val_pids = test_pids 
            train_pids = [p for p in all_pids if p not in test_pids]

        for pid, data in patient_data.items():
            p_inputs = sorted(data['inputs'])
            p_targets = sorted(data['targets'])
            num_slices = len(p_inputs)
            
            for i in range(num_slices):
                target_single = p_targets[i]
                
                if self.context:
                    prev_idx = max(0, i - 1)
                    next_idx = min(num_slices - 1, i + 1)
                    input_data = [p_inputs[prev_idx], p_inputs[i], p_inputs[next_idx]]
                else:
                    input_data = p_inputs[i]
                
                if pid in train_pids:
                    self.train_inputs.append(input_data)
                    self.train_targets.append(target_single)
                if pid in val_pids:
                    self.val_inputs.append(input_data)
                    self.val_targets.append(target_single)
                if pid in test_pids:
                    self.test_inputs.append(input_data)
                    self.test_targets.append(target_single)

    def _get_dataset(self):
        if self.mode == 'train':
            if not self.train_inputs:
                raise ValueError("Train dataset is empty! Check your patient PID configuration.")
            
            train_dataset = Custom_Dataset(
                input_list=self.train_inputs, target_list=self.train_targets, 
                normalize=self.normalize, mode='train', 
                patch_size=self.patch_size, patch_num=self.patch_num, context=self.context
            )
            val_dataset = Custom_Dataset(
                input_list=self.val_inputs, target_list=self.val_targets, 
                normalize=self.normalize, mode='test', context=self.context
            )
            return train_dataset, val_dataset

        elif self.mode == 'test': 
            test_dataset = Custom_Dataset(
                input_list=self.test_inputs, target_list=self.test_targets, 
                normalize=self.normalize, mode='test', context=self.context
            )
            return None, test_dataset

    def create_loaders(self):
        train_dataset, val_dataset = self._get_dataset()
        
        train_loader = None
        val_loader = None
        
        if self.mode == 'train':
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, prefetch_factor=self.prefetch_factor, pin_memory=True)
            if val_dataset and len(val_dataset) > 0:
                val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, prefetch_factor=self.prefetch_factor, pin_memory=True)
        else:
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, prefetch_factor=self.prefetch_factor, pin_memory=True)

        print(f"{' Dataset Information ':=^60}")
        print(f"Dataset Name         : {self.use_dataset}")
        if self.mode == 'train':
            print(f"Training Samples     : {len(train_dataset)}")
            print(f"Validation Samples   : {len(val_dataset) if val_dataset else 0}")
        else:
            print(f"Testing Samples      : {len(val_dataset)}")
        print("=" * 60 + "\n")
        return train_loader, val_loader
