import torch
from skimage.metrics import structural_similarity as ssim
import numpy as np


def transfer_calculate_window(img, dataset_type, MIN_B=-1024, MAX_B=3072, cut_min=-1000, cut_max=1000):
    if dataset_type == "CT":
        # transfer to [-1000, 1000] HU for CT calculate
        img = img * (MAX_B - MIN_B) + MIN_B
        img[img < cut_min] = cut_min
        img[img > cut_max] = cut_max
        img = 255 * (img - cut_min) / (cut_max - cut_min)
    else:
        # full window for PET calculate
        img = img * 255.0
    return img


def transfer_display_window(img, dataset_type, MIN_B=-1024, MAX_B=3072, cut_min=-160, cut_max=240):
    if dataset_type == "CT":
        # transfer to [-160, 240] HU for CT display
        img = img * (MAX_B - MIN_B) + MIN_B
        img[img < cut_min] = cut_min
        img[img > cut_max] = cut_max
        img = 255 * (img - cut_min) / (cut_max - cut_min)
    else:
        # full window for PET display
        img = img * 255.0
    return img


def compute_measure(y, pred, use_dataset):

    if use_dataset in ["mayo16", "mayo20", "local_site1", "local_site2"]:
        dataset_type = "CT"
    else:
        dataset_type = "PET"

    y = transfer_calculate_window(y, dataset_type).squeeze()
    pred = transfer_calculate_window(pred, dataset_type).squeeze()

    psnr = compute_PSNR(pred, y)
    ssim = compute_SSIM(pred, y)
    rmse = compute_RMSE(pred, y, dataset_type)

    return psnr, ssim, rmse


def compute_MSE(img1, img2):

    return ((img1/1.0 - img2/1.0) ** 2).mean()


def compute_RMSE(img1, img2, dataset_type):

    if dataset_type == "CT":
        img1 = img1 * 2000 / 255 + 1000
        img2 = img2 * 2000 / 255 + 1000

    if type(img1) == torch.Tensor:
        return torch.sqrt(compute_MSE(img1, img2)).item()
    else:
        return np.sqrt(compute_MSE(img1, img2))


def compute_PSNR(img1, img2):

    mse = torch.mean((img1 - img2) ** 2)
    max_pixel = img2.max()
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))

    return psnr.item()


def compute_SSIM(img1, img2):

    ssim_s = ssim(img1.numpy(), img2.numpy(), data_range=np.max(img1.numpy()) - np.min(img1.numpy()))

    return ssim_s
