import torch
import piq
from thop import profile, clever_format
from skimage.metrics import structural_similarity as ssim
import numpy as np

def transfer_display_window(img, data_type, MIN_B=-1024, MAX_B=3072, cut_min=-160, cut_max=240):
    if data_type == "CT":
        img = img * (MAX_B - MIN_B) + MIN_B
        img[img < cut_min] = cut_min
        img[img > cut_max] = cut_max
        img = 255 * (img - cut_min) / (cut_max - cut_min)
    else:
        img = torch.clip(img, 0.0, 1.0) * 255.0
    return img

def compute_measure_piq(y, pred, data_range=255.0):
    y = torch.clamp(y.float(), 0.0, 1.0) * data_range
    pred = torch.clamp(pred.float(), 0.0, 1.0) * data_range
    return {
        'psnr': piq.psnr(pred, y, data_range=data_range, reduction='mean').item(),
        'ssim': piq.ssim(pred, y, data_range=data_range, downsample=False, reduction='mean').item(),
        'vif' : piq.vif_p(pred, y, data_range=data_range, reduction='mean').item(),
        'fsim': piq.fsim(pred, y, data_range=data_range, chromatic=False, reduction='mean').item(),
        'rmse': torch.sqrt(torch.mean((pred - y) ** 2)).item(),
    }

def analyze_model_complexity(model, image_size=512, train_patch_size=128, in_channels=1, device='cuda'):
    infer_shape = (1, in_channels, image_size, image_size)
    train_shape = (1, in_channels, train_patch_size, train_patch_size)
    
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    
    infer_data = torch.randn(infer_shape).to(device)
    flops, params = profile(model, inputs=(infer_data, infer_data, infer_data), verbose=False)
    flops_formatted, params_formatted = clever_format([flops, params], "%.3f")
    del infer_data
    
    model.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    
    train_data = torch.randn(train_shape).to(device)
    outputs = model(train_data, train_data, train_data)
    
    loss = outputs[0].mean() if isinstance(outputs, (tuple, list)) else outputs.mean()
    loss.backward()
    
    max_memory = torch.cuda.max_memory_allocated(device)
    memory_formatted = f"{max_memory / 1024 / 1024:.2f} MB"
    model.eval() 

    print(f"{' Model Complexity Analysis ':=^60}")
    print(f"Total Parameters     : {total_params / 1e6:.3f} M") 
    print(f"Inference Shape      : {infer_shape}")
    print(f"FLOPs (Inference)    : {flops_formatted}")
    print(f"Train Patch Shape    : {train_shape}")
    print(f"Training Peak Memory : {memory_formatted}") 
    print("=" * 60 + "\n")

def transfer_calculate_window(img, MIN_B=-1024, MAX_B=3072, cut_min=-1000, cut_max=1000):
    img = img * (MAX_B - MIN_B) + MIN_B
    img[img < cut_min] = cut_min
    img[img > cut_max] = cut_max
    img = 255 * (img - cut_min) / (cut_max - cut_min)
    return img

def compute_measure(y, pred):
    y = transfer_calculate_window(y).squeeze()
    pred = transfer_calculate_window(pred).squeeze()
    return {
        'psnr': compute_PSNR(pred, y), 
        'ssim': compute_SSIM(pred, y), 
        'rmse': compute_RMSE(pred, y),
    }

def compute_MSE(img1, img2):
    return ((img1 / 1.0 - img2 / 1.0) ** 2).mean()

def compute_RMSE(img1, img2):
    img1 = img1 / 255.0 * 2000.0
    img2 = img2 / 255.0 * 2000.0
    return torch.sqrt(compute_MSE(img1, img2)).item()

def compute_PSNR(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    max_pixel = img2.max()
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()

def compute_SSIM(img1, img2):
    ssim_s = ssim(img1.cpu().numpy(), img2.cpu().numpy(), data_range=np.max(img1.cpu().numpy()) - np.min(img1.cpu().numpy()))
    return ssim_s

