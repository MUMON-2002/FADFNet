import torch
from torch.nn import functional as F
import numpy as np
import os
import importlib
import time

from utils.transform import WaveletTransformer
from utils.dataloader import LoaderHook
from utils.options import TrainOptions
from utils.measure import compute_measure
from utils.lr_schedule import HybridLRScheduler
from utils.model_complexity import analyze_model_complexity, print_model_complexity

np.random.seed(392)
torch.manual_seed(392)

class Trainer:

    def __init__(self, opt):

        self.opt = opt
        self.device = torch.device(f'cuda:{opt.gpu_id}' if torch.cuda.is_available() else 'cpu')

        self.model = self._create_model()
        self.WaveletTransformer = WaveletTransformer(wave_level=opt.wave_level, wavelet_type=opt.wave_type)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.opt.learning_rate,
            weight_decay=3e-4,
            betas=(0.9, 0.999)
        )

        self.start_epoch = self._load_checkpoint()
        self.use_dataset = self.opt.use_dataset

        self.dataloader_hook = LoaderHook(
            mode='train', 
            use_dataset=self.use_dataset, 
            num_workers=16,
            prefetch_factor=4,
            switch_epochs=[0, 5, 10, 15], 
            switch_bs=[opt.batch_size, opt.batch_size, opt.batch_size, opt.batch_size], 
            switch_ps=[64, 128, 256, 512], 
            switch_pn=[64, 16, 4, 1],
            train_ratio=opt.train_ratio,
        )

        self.train_data_loader, self.val_data_loader = self.dataloader_hook.get_data_loader(self.start_epoch)

        self.iters_per_epoch = len(self.train_data_loader)

        self.max_epoch = self.opt.total_epoch
        self.warmup_epoch = self.opt.warmup_epoch

        self.max_iters = self.max_epoch * self.iters_per_epoch
        self.warmup_iters = self.warmup_epoch * self.iters_per_epoch

        self.scheduler = HybridLRScheduler(
            self.optimizer,
            total_iters=self.max_iters,
            warmup_iters=self.warmup_iters,
            eta_min=1e-6,
            last_epoch=-1
        )

        if not os.path.exists(self.opt.path_to_save):
            os.makedirs(self.opt.path_to_save)

        self.train_loss_history = []

    def _create_model(self):

        self.model_name = self.opt.which_model
        module = importlib.import_module(f'model.{self.model_name}')
        
        model = module.XNet(
            inp_channels=1,
            out_channels=1,
            dim=self.opt.dim,
            activation=self.opt.activation,
            norm_type=self.opt.norm_type,
        ).to(self.device)

        return model

    def _load_checkpoint(self):

        if self.opt.continue_to_train:
            ckpt_model = torch.load(self.opt.ckpt_path, map_location=self.device)
            self.model.load_state_dict(ckpt_model["net_model"])
            self.optimizer.load_state_dict(ckpt_model["optimG"])
            start_epoch = ckpt_model["epoch"] + 1
            
            print(f"Resuming training from epoch {start_epoch}")
            return start_epoch
        
        return 0
    
    def _analyze_complexity(self, patch_size=512):
        
        input_shape = (1, 1, patch_size, patch_size)
        
        results = analyze_model_complexity(
            self.model, 
            input_shape=input_shape, 
            device=self.device
        )
    
        print_model_complexity(results, input_shape)
        return results

    def _compute_losses(self, prediction, target):

        pixel_loss = F.l1_loss(prediction, target)
        total_loss = pixel_loss

        return total_loss   
    
    def _train_epoch(self, epoch):

        epoch_start = time.time()
        self.model.train()

        epoch_train_loss = 0.0
        total_samples = 0

        for batch_idx, (data, target) in enumerate(self.train_data_loader):
            batch_size, patch_num, dim, patch_size, patch_size = data.shape

            data = data.float().view(-1, dim, patch_size, patch_size).to(self.device)
            target = target.float().view(-1, dim, patch_size, patch_size).to(self.device)

            data_low, data_high = self.WaveletTransformer(data)
            recon = self.model(data_low, data_high)
            
            total_loss = self._compute_losses(recon, target) 
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
            self.scheduler.step()

            epoch_train_loss += total_loss.item() * batch_size
            total_samples += batch_size

        avg_train_loss = epoch_train_loss / total_samples
        self.train_loss_history.append(avg_train_loss)
        epoch_time = time.time() - epoch_start
        current_lr = self.scheduler.get_last_lr()[0]

        print(f"Epoch {epoch} Summary ({epoch_time:.1f}s): "
            f"Learning Rate: {current_lr:.2e} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Train Samples: {total_samples}")

    def _validate_epoch(self, epoch):

        if (epoch + 1) % self.opt.validation_freq != 0 or self.opt.train_ratio == 1.0:
            return

        self.model.eval()
        psnr_vals, ssim_vals, rmse_vals = [], [], []

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.val_data_loader):
                batch_size, patch_num, dim, patch_size, patch_size = data.shape

                data = data.float().view(-1, dim, patch_size, patch_size).to(self.device)
                target = target.float().view(-1, dim, patch_size, patch_size).to(self.device)

                data_low, data_high = self.WaveletTransformer(data)
                recon = self.model(data_low, data_high)

                val_psnr, val_ssim, val_rmse = compute_measure(target.detach().cpu(), recon.detach().cpu(), self.use_dataset)

                psnr_vals.extend([val_psnr] * batch_size)
                ssim_vals.extend([val_ssim] * batch_size)
                rmse_vals.extend([val_rmse] * batch_size)

        avg_psnr = np.mean(psnr_vals)
        avg_ssim = np.mean(ssim_vals)
        avg_rmse = np.mean(rmse_vals)

        var_psnr = np.var(psnr_vals, ddof=1)
        var_ssim = np.var(ssim_vals, ddof=1)
        var_rmse = np.var(rmse_vals, ddof=1)

        print(f"Epoch {epoch} Validation Results:")
        print(f" PSNR: {avg_psnr:.2f} (Var: {var_psnr:.4f}) | "
            f" SSIM: {avg_ssim:.4f} (Var: {var_ssim:.6f}) | "
            f" RMSE: {avg_rmse:.2f} (Var: {var_rmse:.4f}) | "
            f" Val samples: {len(self.val_data_loader)}")


    def _save_checkpoint(self, epoch):

        if (epoch + 1) % self.opt.save_freq == 0:
            ckpt = {
                "net_model": self.model.state_dict(),
                "optimG": self.optimizer.state_dict(),
                "epoch": epoch
            }

            save_path = os.path.join(self.opt.path_to_save, f"model_epoch_{epoch + 1}.pkl")
            torch.save(ckpt, save_path)
            print(f"Model saved at {save_path}")
    
    def train(self):

        for epoch in range(self.start_epoch, self.max_epoch):
            
            self.train_data_loader, self.val_data_loader = self.dataloader_hook.get_data_loader(epoch)

            self._train_epoch(epoch)
            self._validate_epoch(epoch)
            self._save_checkpoint(epoch)


def main():

    opt = TrainOptions().parse()
    trainer = Trainer(opt)
    trainer._analyze_complexity(patch_size=512)
    trainer.train()


if __name__ == "__main__":

    main()

