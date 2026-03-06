import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import importlib
import time
import pandas as pd

from data.dataloader import Loader_Hook
from data.dataset_info import DATASET_INFO

from utils.options import TrainOptions
from utils.measure import compute_measure, analyze_model_complexity
from utils.lr_schedule import HybridLRScheduler
from utils.decomposition import DTCWT_Transformer
from utils.loss_fn import *

np.random.seed(42)
torch.manual_seed(42)

class Trainer:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device(f'cuda:{opt.gpu_id}' if torch.cuda.is_available() else 'cpu')

        self.start_epoch = self._load_checkpoint()
        self.use_dataset = self.opt.use_dataset
        self.data_type = DATASET_INFO[self.use_dataset]["data_type"]
        self.context = self.opt.context
        self.model = self._create_model()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=opt.learning_rate,
            weight_decay=3e-4,
            betas=(0.9, 0.999)
        )
        
        self.loader_hook = Loader_Hook(
            mode='train', 
            use_dataset=self.use_dataset, 
            num_workers=16,
            prefetch_factor=4,
            batch_size=opt.batch_size,    
            patch_size=opt.patch_size,
            patch_num=opt.patch_num,
            context=opt.context,
            set_val=opt.set_val
        )

        self.train_data_loader, self.val_data_loader = self.loader_hook.create_loaders()
        self.iters_per_epoch = len(self.train_data_loader)
        self.patch_size = opt.patch_size
        self.patch_num = opt.patch_num

        self.lambda_recon = 1.0
        self.lambda_low = opt.lambda_low
        self.lambda_high = opt.lambda_high
        self.lambda_ssim = opt.lambda_ssim

        self.pixel_loss = CharbonnierLoss().to(self.device)
        self.ssim_loss = SSIMLoss().to(self.device)

        self.wt = DTCWT_Transformer(wave_level=opt.wave_level)

        self.max_epoch = opt.total_epoch
        self.warmup_epoch = opt.warmup_epoch
        self.max_iters = self.max_epoch * self.iters_per_epoch
        self.warmup_iters = self.warmup_epoch * self.iters_per_epoch

        self.scheduler = HybridLRScheduler(
            self.optimizer,
            total_iters=self.max_iters,
            warmup_iters=self.warmup_iters,
            eta_min=1e-6,
            last_epoch=-1
        )
        
        if not os.path.exists(opt.path_to_save):
            os.makedirs(opt.path_to_save)

        in_channels = 3 if self.context else 1
        analyze_model_complexity(
            self.model, 
            image_size=512, 
            train_patch_size=self.patch_size, 
            in_channels=in_channels, 
            device=self.device
        )

    def _create_model(self):
        self.model_name = self.opt.which_model
        module = importlib.import_module(f'model.{self.model_name}')
        in_channels = 3 if self.context else 1
        model = module.Denoising_Network(img_channels=in_channels, dim=self.opt.dim).to(self.device)
        return model

    def _load_checkpoint(self):
        if self.opt.continue_to_train:
            ckpt_model = torch.load(self.opt.ckpt_path, map_location=self.device)
            self.model.load_state_dict(ckpt_model["net_model"])
            self.optimizer.load_state_dict(ckpt_model["optimG"])
            start_epoch = ckpt_model["epoch"] + 1
            
            print(f"Resuming training from epoch {start_epoch}")
            return start_epoch
        return 1
    
    def _compute_loss(self, recon, target, recon_low, recon_high, target_low, target_high):
        l_recon = self.pixel_loss(recon, target) 
        l_ssim = self.ssim_loss(recon, target)
        l_low = self.pixel_loss(recon_low, target_low)
        l_high = self.pixel_loss(recon_high, target_high)

        total_loss = self.lambda_recon * l_recon +  self.lambda_low * l_low + self.lambda_high * l_high + self.lambda_ssim * l_ssim
        losses = {'recon': l_recon, 'low': l_low, 'high': l_high, 'ssim':l_ssim}
        return total_loss, losses
    
    def _train_epoch(self, epoch):  
        epoch_start = time.time()
        self.model.train()

        epoch_losses_sum = {}
        epoch_losses_sum['total'] = 0.0
        total_samples = 0

        for batch_idx, (data, target) in enumerate(self.train_data_loader):
            data = data.view(-1, data.size(-3), data.size(-2), data.size(-1))
            target = target.view(-1, target.size(-3), target.size(-2), target.size(-1))
            actual_batch_size = data.shape[0]

            data = data.float().to(self.device)
            target = target.float().to(self.device)

            data_low, data_high = self.wt(data)
            target_low, target_high = self.wt(target)

            recon, recon_low, recon_high = self.model(data_low, data_high, data)
            total_loss, losses = self._compute_loss(recon, target, recon_low, recon_high, target_low, target_high)

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_samples += actual_batch_size
            epoch_losses_sum['total'] += total_loss.item() * actual_batch_size

            for loss_name, loss_value in losses.items():
                if loss_name not in epoch_losses_sum:
                    epoch_losses_sum[loss_name] = 0.0
                epoch_losses_sum[loss_name] += loss_value.item() * actual_batch_size

        avg_losses = {}
        for loss_name, loss_sum in epoch_losses_sum.items():
            avg_losses[loss_name] = loss_sum / total_samples

        epoch_time = time.time() - epoch_start
        current_lr = self.scheduler.get_last_lr()[0]

        log_msgs = []
        log_msgs.append(f"Epoch {epoch} Summary ({epoch_time:.1f}s): ")
        log_msgs.append(f"LR: {current_lr:.2e} | ")
        log_msgs.append(f"Total Loss: {avg_losses['total']:.2e} | ")
            
        for loss_name, avg_value in sorted(avg_losses.items()):
            if loss_name == 'total': continue
            log_msgs.append(f"{loss_name.capitalize()}: {avg_value:.2e} | ")
        print("".join(log_msgs))
        
    def _validate_epoch(self, epoch):
        if not self.val_data_loader: 
            return
        if epoch % self.opt.validation_freq != 0 and epoch != -1 and epoch != self.max_epoch: 
            return

        self.model.eval()
        val_results = []

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.val_data_loader):
                data = data.float().to(self.device)
                target = target.float().to(self.device)

                if epoch == -1:
                    baseline_data = data[:, 1:2, :, :] if data.shape[1] == 3 else data
                    batch_metrics = compute_measure(target, baseline_data)
                else:
                    data_low, data_high = self.wt(data)
                    recon, _, _ = self.model(data_low, data_high, data)
                    batch_metrics = compute_measure(target, recon)
                
                val_results.append(batch_metrics)

        if len(val_results) > 0:
            df = pd.DataFrame(val_results)
            avg = df.mean()
            
            prefix = "BASELINE" if epoch == -1 else f"EPOCH {epoch}"
            title = f" Validation Results: {prefix} "
            
            print(f"{title:-^60}")
            print(f"PSNR: {avg['psnr']:.2f}  |  SSIM: {avg['ssim']:.4f}  |  RMSE: {avg['rmse']:.4f}")
            print("-" * 60 + "\n")

    def _save_checkpoint(self, epoch):
        if epoch % self.opt.save_freq == 0 or epoch == self.max_epoch:
            ckpt = {
                "net_model": self.model.state_dict(),
                "optimG": self.optimizer.state_dict(),
                "epoch": epoch
            }
            save_path = os.path.join(self.opt.path_to_save, f"model_epoch_{epoch}.pkl")
            torch.save(ckpt, save_path)
            print(f"Model saved at {save_path}")

    def train(self):
        print(f"{' Training Started ':=^60}")
        self._validate_epoch(epoch=-1)

        for epoch in range(self.start_epoch, self.max_epoch + 1):
            self._train_epoch(epoch)
            self._validate_epoch(epoch)
            self._save_checkpoint(epoch)
        print(f"{' Training Completed ':=^60}")

def main():
    opt = TrainOptions().parse()
    trainer = Trainer(opt)
    trainer.train()

if __name__ == "__main__":
    main()