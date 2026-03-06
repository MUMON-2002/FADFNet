import torch
import numpy as np
from PIL import Image
import os
import importlib
from tqdm import tqdm
import random
import pandas as pd

from utils.options import TestOptions
from data.dataloader import Loader_Hook
from utils.measure import compute_measure, transfer_display_window
from data.dataset_info import DATASET_INFO
from utils.decomposition import DTCWT_Transformer

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

class Tester:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device(f'cuda:{opt.gpu_id}' if torch.cuda.is_available() else 'cpu')
        
        self.use_dataset = opt.use_dataset
        self.data_type = DATASET_INFO[self.use_dataset]["data_type"]
        self.context = self.opt.context

        self.loader_hook = Loader_Hook(
            mode='test',
            use_dataset=self.use_dataset,
            num_workers=16,
            prefetch_factor=4,
            context=self.context 
        )
        
        _, self.val_data_loader = self.loader_hook.create_loaders()
        self.wt = DTCWT_Transformer(wave_level=opt.wave_level).to(self.device).eval()

        self.output_root = opt.output_root
        os.makedirs(self.output_root, exist_ok=True)
        self.model = self._create_and_load_model()

    def _create_and_load_model(self):
        self.model_name = self.opt.which_model
        module = importlib.import_module(f'model.{self.model_name}')
        
        in_channels = 3 if self.context else 1
        model = module.Denoising_Network(img_channels=in_channels, dim=self.opt.dim).to(self.device)
        
        ckpt_model = torch.load(self.opt.ckpt_path, map_location=self.device, weights_only=True)
        state_dict = ckpt_model["net_model"]

        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        state_dict = {k: v for k, v in state_dict.items() if not k.endswith(('total_ops', 'total_params'))}
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def _save_batch_images(self, data, recon, target, batch_idx):
        if data.shape[1] == 3:
            data = data[:, 1:2, :, :]
            
        data_display = transfer_display_window(data, self.data_type)
        recon_display = transfer_display_window(recon, self.data_type)
        target_display = transfer_display_window(target, self.data_type)

        data_path = os.path.join(self.output_root, "images", f"input_{batch_idx:04d}.png")
        recon_path = os.path.join(self.output_root, "images", f"recon_{batch_idx:04d}.png")
        target_path = os.path.join(self.output_root, "images", f"target_{batch_idx:04d}.png")
        
        Image.fromarray(data_display[0, 0].cpu().numpy().astype(np.uint8)).save(data_path)
        Image.fromarray(recon_display[0, 0].cpu().numpy().astype(np.uint8)).save(recon_path)
        Image.fromarray(target_display[0, 0].cpu().numpy().astype(np.uint8)).save(target_path)

    def test(self):
        metrics_vals = []
        origin_metrics_vals = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(self.val_data_loader, desc="Testing")):
                data = data.float().to(self.device)
                target = target.float().to(self.device)
                
                baseline_data = data[:, 1:2, :, :] if data.shape[1] == 3 else data
                
                data_low, data_high = self.wt(data)
                recon, _, _ = self.model(data_low, data_high, data)
                
                self._save_batch_images(data, recon, target, batch_idx)
                
                metrics_vals.append(compute_measure(target, recon))
                origin_metrics_vals.append(compute_measure(target, baseline_data))

        df_denoised = pd.DataFrame(metrics_vals)
        df_original = pd.DataFrame(origin_metrics_vals)

        # ================= Save to Excel =================
        df_final = pd.concat([df_denoised.add_prefix('Denoised_'), df_original.add_prefix('Original_')], axis=1)
        
        excel_path = os.path.join(self.output_root, 'test_metrics.xlsx')
        try:
            df_final.to_excel(excel_path, index_label='Sample_Idx')
            print(f"Metrics Excel saved to : {excel_path}")
        except ImportError:
            print("Saving Excel failed. Please install openpyxl: pip install openpyxl")
            df_final.to_csv(excel_path.replace('.xlsx', '.csv'), index_label='Sample_Idx')
        # =================================================
        
        avg_denoised, var_denoised  = df_denoised.mean(), df_denoised.std(ddof=1)
        avg_original, var_original = df_original.mean(), df_original.std(ddof=1)

        print(f"\n{' Test Results':=^60}")

        print("DENOISED (Model Output vs. Target):")
        print(f"   PSNR:  {avg_denoised['psnr']:.2f} (Var: {var_denoised['psnr']:.4f})")
        print(f"   SSIM:  {avg_denoised['ssim']:.4f} (Var: {var_denoised['ssim']:.6f})")
        print(f"   RMSE:  {avg_denoised['rmse']:.4f} (Var: {var_denoised['rmse']:.6f})")
        print("-" * 60)

        print("ORIGINAL (Input vs. Target):")
        print(f"   PSNR:  {avg_original['psnr']:.2f} (Var: {var_original['psnr']:.4f})")
        print(f"   SSIM:  {avg_original['ssim']:.4f} (Var: {var_original['ssim']:.6f})")
        print(f"   RMSE:  {avg_original['rmse']:.4f} (Var: {var_original['rmse']:.6f})")
        print("=" * 60)

def main():
    opt = TestOptions().parse()   
    tester = Tester(opt)  
    tester.test()

if __name__ == "__main__":
    main()