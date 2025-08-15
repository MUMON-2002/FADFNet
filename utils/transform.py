import numpy as np
from PIL import Image
import os
from glob import glob
import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward, DWTInverse


class WaveletTransformer(nn.Module):

    def __init__(self, wavelet_type="haar", wave_level=1):
        super().__init__()

        self.wavelet_type = wavelet_type
        self.wave_level = wave_level

        self.dwt = DWTForward(J=self.wave_level, wave=wavelet_type, mode='symmetric')
        self.idwt = DWTInverse(wave=wavelet_type, mode='symmetric')

    def forward(self, x):

        device = x.device
        self.dwt = self.dwt.to(device)
        self.idwt = self.idwt.to(device)
        yl, yh = self.dwt(x)

        zero_yl = torch.zeros_like(yl).to(device)
        zero_yh = []

        for i in range(self.wave_level):

            zero_y = torch.zeros_like(yh[i]).to(device)
            zero_yh.append(zero_y)

        x_low = self.idwt((yl, zero_yh))
        x_high = self.idwt((zero_yl, yh))

        return x_low, x_high


class FFTTransformer(nn.Module):

    def __init__(self, ratio=0.1, gaussian=False, sigma_factor=1.0):
        super(FFTTransformer, self).__init__()

        self.ratio = ratio
        self.gaussian = gaussian
        self.sigma_factor = sigma_factor

    def forward(self, x):

        fft = torch.fft.fft2(x, dim=(-2, -1))
        fft_shift = torch.fft.fftshift(fft, dim=(-2, -1))

        mask = self._get_mask(x)
        low_fft = fft_shift * mask
        high_fft = fft_shift * (1 - mask)

        low_fft_ishift = torch.fft.ifftshift(low_fft, dim=(-2, -1))
        high_fft_ishift = torch.fft.ifftshift(high_fft, dim=(-2, -1))

        low_freq = torch.fft.ifft2(low_fft_ishift, dim=(-2, -1)).real
        high_freq = torch.fft.ifft2(high_fft_ishift, dim=(-2, -1)).real

        return low_freq, high_freq

    def _get_mask(self, x):

        B, C, H, W = x.shape
        device = x.device

        center_h, center_w = H // 2, W // 2
        cutoff_radius = min(H, W) * self.ratio / 2

        h_coords = torch.arange(H, device=device).view(-1, 1)
        w_coords = torch.arange(W, device=device)

        distance = torch.sqrt((h_coords - center_h)**2 + (w_coords - center_w)**2)

        if self.gaussian:
            sigma = cutoff_radius * self.sigma_factor
            mask = torch.exp(-distance**2 / (2 * sigma**2))
        else:
            mask = (distance <= cutoff_radius).float()

        mask = mask.unsqueeze(0).unsqueeze(0)

        return mask