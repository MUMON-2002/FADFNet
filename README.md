# FADFNet

This repository contains the official PyTorch implementation of the paper: 
**[FADFNet: A Fine-tunable and Adaptive Decomposition-Fusion Network for Cross-dataset Low-dose CT and Low-dose PET Image Reconstruction](https://doi.org/10.1016/j.media.2026.104016)**.

## 1. Environment Setup

First, create the Conda environment using the provided `environment.yml` file and activate it:

```bash
cd FADFNet
conda env create -f environment.yml
conda activate fadfnet
```

## 2. Data Preparation

To train or test the FADFNet model, you need to prepare your dataset by following these steps:

1. **Format & Normalize**: Normalize your data files (`.nii` or `.dicom`) to the range of `(0, 1.0)` and save them as numpy files.
2. **Naming Convention**: Name the processed files as `(patient_number)_(slice_number)_(data_type).npy`.
3. **Configuration**: Modify the `DATASET_INFO` dictionary in `data/dataset_info.py`  to configure your dataset paths and test splits. Example:

```python
DATASET_INFO = {
    "dataset_name": {
        "dir": "DATASET_PATH",
        "input_label": "*_input.npy", 
        "target_label": "*_target.npy",
        "test_patients": ['test_id'],
        "val_patients": ["val_id"],
        "data_type": "CT",
    },
    # Add other datasets here...
}
```

## 3. Training

Run the following command to start training the model from scratch. You can adjust the parameters as needed. Example:

```bash
python train.py \
    --gpu_id 1 \
    --path_to_save checkpoints/exp/ \
    --use_dataset YOUR_DATASET_NAME	 \
    --batch_size 8 \
    --warmup_epoch 20 \
    --total_epoch 100 \
    --wave_level 2 \
    --context
```

## 4. Testing

To evaluate the model using a pre-trained checkpoint, run the test script. Example:

```bash
python test.py \
    --gpu_id 0 \
    --ckpt_path checkpoints/exp/model_epoch_X.pkl \
    --use_dataset YOUR_DATASET_NAME \
    --wave_level 2 \
    --output_root results/exp \
    --context
```

## Citation

If you find this code or our paper useful for your research, please consider citing:

```bibtex
@article{FADFNet2026,
	title = {FADFNet: A Fine-tunable and Adaptive Decomposition-Fusion Network for Cross-dataset Low-dose CT and Low-dose PET Image Reconstruction},
	author = {Fangji Qian and Weitao Wang and Yanyan Huang and Meng Niu and Yuanxue Gao and Zihao Zhao and Kuangyu Shi and Lequan Yu and Yu Fu and Cheng Zhuo},
	journal = {Medical Image Analysis},
	pages = {104016},
	year = {2026},
	issn = {1361-8415},
}
```

