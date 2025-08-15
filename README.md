# FADFNet: Fine-tunable Adaptive Decomposition-Fusion Network

## Introduction

This repository contains an early implementation of **FADFNet: A Fine-tunable and Adaptive Decomposition-Fusion Network** for **Cross-dataset Low-dose CT and Low-dose PET Image Reconstruction**.

## Setup

### Environment Setup



### Data Preparation

1. Normalize your data files (.nii or .dicom) to the range of (0, 1.0) and save them as numpy files, named as `(patient_number)_(slice_number)_(data_type).npy`.
2. Modify the `DATASET_INFO` dictionary in `./utils/dataloader.py` to configure your dataset as follows:

```
DATASET_INFO = {
    "dataset_name": {
        "dir": "dataset_path",
        "input_label": "*_input.npy", 
        "target_label": "*_target.npy",
        "test_patients": ['test_id'],
    },
    ...
}
```

### Training

To train the model, run the following command:

```
python train.py --gpu_id 0 --path_to_save ./checkpoints/
```

### Testing

For testing, use:

```
python test.py --gpu_id 0 --output_root ./results --ckpt_path ./checkpoints/model_epoch_5.pkl
```

