import torch
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
from utils.options import TestOptions
from utils.dataloader import LoaderHook 
from utils.transform import WaveletTransformer
from utils.measure import compute_measure, transfer_display_window
import importlib
from typing import List, Tuple, Dict


class Tester:
    
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device(f'cuda:{opt.gpu_id}' if torch.cuda.is_available() else 'cpu')
        
        self.model = self._load_model()
        self.WaveletTransformer = WaveletTransformer(wave_level=opt.wave_level, wavelet_type=opt.wave_type)

        self.use_dataset = opt.use_dataset

        self.dataloader_hook = LoaderHook(
            mode='test',
            use_dataset=opt.use_dataset,
            num_workers=16,
            prefetch_factor=4,
            switch_epochs=[0],
            switch_bs=[1],
            switch_ps=[512], 
            switch_pn=[1]
        )
        
        _, self.val_data_loader = self.dataloader_hook.get_data_loader(0)
        
        print(f"Model loaded from: {opt.ckpt_path}")

        self.output_root = opt.output_root
        os.makedirs(self.output_root, exist_ok=True)

    def _load_model(self):
        module = importlib.import_module(f'model.{self.opt.which_model}')
        
        model = module.XNet(
            inp_channels=1,
            out_channels=1,
            dim=self.opt.dim,
            activation=self.opt.activation,
            norm_type=self.opt.norm_type,
        ).to(self.device)
        
        ckpt_model = torch.load(self.opt.ckpt_path, map_location=self.device, weights_only=False)
        
        state_dict = ckpt_model["net_model"]
        state_dict = {k: v for k, v in state_dict.items() 
                            if not k.endswith(('total_ops', 'total_params'))}
        
        model.load_state_dict(state_dict)
        model.eval()
        
        return model


    def _save_batch_images(self, data: torch.Tensor, recon: torch.Tensor, 
                         target: torch.Tensor, batch_idx: int):
        
        data_display = transfer_display_window(data, self.use_dataset)
        recon_display = transfer_display_window(recon, self.use_dataset)
        target_display = transfer_display_window(target, self.use_dataset)
        
        data_path = os.path.join(self.output_root, f"input_{batch_idx}.png")
        recon_path = os.path.join(self.output_root, f"recon_{batch_idx}.png")
        target_path = os.path.join(self.output_root, f"target_{batch_idx}.png")

        Image.fromarray(data_display.squeeze().cpu().numpy().astype(np.uint8)).save(data_path)
        Image.fromarray(recon_display.squeeze().cpu().numpy().astype(np.uint8)).save(recon_path)
        Image.fromarray(target_display.squeeze().cpu().numpy().astype(np.uint8)).save(target_path)

    def test(self) -> Dict[str, float]:
        
        psnr_list = []
        ssim_list = []
        rmse_list = []
        origin_psnr_list = []
        origin_ssim_list = []
        origin_rmse_list = []
        
        for batch_idx, (data, target) in enumerate(tqdm(self.val_data_loader, desc="Testing")):
            batch_size, patch_num, dim, patch_size, patch_size = data.shape

            data = data.float().view(-1, dim, patch_size, patch_size).to(self.device)
            target = target.float().view(-1, dim, patch_size, patch_size).to(self.device)

            data_low, data_high = self.WaveletTransformer(data)
            
            with torch.no_grad():
                recon = self.model(data_low, data_high)
            
            psnr, ssim, rmse = compute_measure(target.detach().cpu(), recon.detach().cpu(), self.use_dataset)
            origin_psnr, origin_ssim, origin_rmse = compute_measure(target.detach().cpu(), data.detach().cpu(), self.use_dataset)
            
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            rmse_list.append(rmse)
            origin_psnr_list.append(origin_psnr)
            origin_ssim_list.append(origin_ssim)
            origin_rmse_list.append(origin_rmse)

            self._save_batch_images(data, recon, target, batch_idx)

        results = {
            'denoised_psnr': np.mean(psnr_list),
            'denoised_ssim': np.mean(ssim_list),
            'denoised_rmse': np.mean(rmse_list),
            'original_psnr': np.mean(origin_psnr_list),
            'original_ssim': np.mean(origin_ssim_list),
            'original_rmse': np.mean(origin_rmse_list),
        }

        print("Test Results:")
        print(f"DENOISED - PSNR: {results['denoised_psnr']:.4f}, "
              f"SSIM: {results['denoised_ssim']:.4f}, "
              f"RMSE: {results['denoised_rmse']:.4f}"
        )
        print(f"ORIGINAL - PSNR: {results['original_psnr']:.4f}, "
              f"SSIM: {results['original_ssim']:.4f}, "
              f"RMSE: {results['original_rmse']:.4f}"
        )
        
        return results


def main():
    opt = TestOptions().parse()   
    tester = Tester(opt)  
    results = tester.test()

if __name__ == "__main__":
    main()
