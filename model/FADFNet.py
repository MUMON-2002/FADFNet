import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        var = x.var(1, keepdim=True, unbiased=False)
        std = (var + self.eps).sqrt()
        x = (x - mean) / std
        x = x * self.weight + self.bias
        return x
    

class Downsample(nn.Module):
    def __init__(self, in_dim):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim, bias=False),
            nn.Conv2d(in_dim, in_dim // 2, kernel_size=1, bias=False),
        )
        self.unshuffle = nn.PixelUnshuffle(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.unshuffle(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_dim):
        super(Upsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim, bias=False),
            nn.Conv2d(in_dim, in_dim * 2, kernel_size=1, bias=False),
        )
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.shuffle(x)
        return x


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class AVG_Attention(nn.Module):
    def __init__(self, dim):
        super(AVG_Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv(x)
        x = self.sigmoid(x)
        return x


class STD_Attention(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super(STD_Attention, self).__init__()
        self.conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, bias=True)
        self.eps = eps
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = (torch.var(x, dim=[2, 3], keepdim=True, unbiased=False) + self.eps).sqrt()
        x = self.conv(x)
        x = self.sigmoid(x)
        return x
    

class FuseBlock(nn.Module):
    def __init__(self, dim, DW_Expand=2, FFN_Expand=2, bias=False):
        super(FuseBlock, self).__init__()
        self.dim = dim
        dw_channel = dim * DW_Expand
        ffn_channel = FFN_Expand * dim
        self.patch_size = 8

        self.norm1 = LayerNorm2d(dim)
        self.to_hidden_1 = nn.Conv2d(dim, dw_channel * 3, kernel_size=1, bias=bias)
        self.to_hidden_dw_1 = nn.Conv2d(dw_channel * 3, dw_channel * 3, kernel_size=3, padding=1, groups=dw_channel * 3, bias=bias)
        self.norm2 = LayerNorm2d(dw_channel)
        self.projection_1 = nn.Conv2d(dw_channel, dim, kernel_size=1, bias=bias)

        self.norm3 = LayerNorm2d(dim)
        self.to_hidden_2 = nn.Conv2d(in_channels=dim, out_channels=ffn_channel, kernel_size=1, padding=0, bias=True)
        self.to_hidden_dw_2 = nn.Conv2d(ffn_channel, ffn_channel, kernel_size=3, padding=1, groups=ffn_channel, bias=bias)
        self.projection_2 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=dim, kernel_size=1, padding=0, bias=False)
        self.sg = SimpleGate()

        self.scale_1 = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.scale_2 = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        B, C, H, W = x.shape
        identity = x
        x = self.norm1(x)

        hidden = self.to_hidden_1(x)
        q, k, v = self.to_hidden_dw_1(hidden).chunk(3, dim=1)

        q_patch = rearrange(q, 'b c (h p1) (w p2) -> b c h w p1 p2', p1=self.patch_size, p2=self.patch_size)
        k_patch = rearrange(k, 'b c (h p1) (w p2) -> b c h w p1 p2', p1=self.patch_size, p2=self.patch_size)
        q_fft = torch.fft.rfft2(q_patch.float())
        k_fft = torch.fft.rfft2(k_patch.float())
        attn_fft = q_fft * k_fft

        attn = torch.fft.irfft2(attn_fft, s=(self.patch_size, self.patch_size))
        attn = rearrange(attn, 'b c h w p1 p2 -> b c (h p1) (w p2)', p1=self.patch_size, p2=self.patch_size)
        attn = self.norm2(attn)
        
        out = v * attn
        out = self.projection_1(out)
        x = identity + out * self.scale_1 

        identity = x
        x = self.norm3(x)
        x = self.to_hidden_2(x)
        x = self.to_hidden_dw_2(x)
        x = self.sg(x)
        x = self.projection_2(x)
        x = identity + x * self.scale_2 
        return x


class FAEBlock(nn.Module):
    def __init__(self, dim, attn_module=None, DW_Expand=2, FFN_Expand=2):
        super(FAEBlock, self).__init__()
        self.dim = dim
        dw_channel = dim * DW_Expand
        ffn_channel = FFN_Expand * dim
        
        # --- Conv Block ---
        self.norm1 = LayerNorm2d(dim)
        self.to_hidden_1 = nn.Conv2d(in_channels=dim, out_channels=dw_channel, kernel_size=1, padding=0, bias=False)
        self.to_hidden_dw_1 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, groups=dw_channel, bias=True)
        self.projection_1 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=dim, kernel_size=1, padding=0, bias=False)
        
        self.sg = SimpleGate()
        self.attn = attn_module

        # --- FFN Block ---
        self.norm2 = LayerNorm2d(dim)
        self.to_hidden_2 = nn.Conv2d(in_channels=dim, out_channels=ffn_channel, kernel_size=1, padding=0, bias=True)
        self.projection_2 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=dim, kernel_size=1, padding=0, bias=False)

        self.scale_1 = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.scale_2 = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        identity = x

        x = self.norm1(x)
        x = self.to_hidden_1(x)
        x = self.to_hidden_dw_1(x)
        x = self.sg(x)
        if self.attn is not None:
            x = x * self.attn(x)
        x = self.projection_1(x)
        x = identity + x * self.scale_1

        identity = x
        x = self.norm2(x)
        x = self.to_hidden_2(x)
        x = self.sg(x)
        x = self.projection_2(x)
        x = identity + x * self.scale_2
        return x
    
    
class LFBlocks(nn.Module):
    def __init__(self, dim, num_blocks, DW_Expand=2, FFN_Expand=2):
        super(LFBlocks, self).__init__()
        self.layers = nn.ModuleList()
        dw_channel = dim * DW_Expand
        for _ in range(num_blocks):
            attn_module = AVG_Attention(dim=dw_channel // 2)
            self.layers.append(FAEBlock(dim=dim, attn_module=attn_module, DW_Expand=DW_Expand, FFN_Expand=FFN_Expand))
    
    def forward(self, x):
        for _, layer in enumerate(self.layers):
            x = layer(x)
        return x


class HFBlocks(nn.Module):
    def __init__(self, dim, num_blocks, DW_Expand=2, FFN_Expand=2):
        super(HFBlocks, self).__init__()
        self.layers = nn.ModuleList()
        dw_channel = dim * DW_Expand
        for _ in range(num_blocks):
            attn_module = STD_Attention(dim=dw_channel // 2)
            self.layers.append(FAEBlock(dim=dim, attn_module=attn_module, DW_Expand=DW_Expand, FFN_Expand=FFN_Expand))
    
    def forward(self, x):
        for _, layer in enumerate(self.layers):
            x = layer(x)
        return x


class FuseBlocks(nn.Module):
    def __init__(self, dim, num_blocks, DW_Expand=2, FFN_Expand=2):
        super(FuseBlocks, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_blocks):
            self.layers.append(FuseBlock(dim=dim, DW_Expand=DW_Expand, FFN_Expand=FFN_Expand))
    
    def forward(self, x):
        for _, layer in enumerate(self.layers):
            x = layer(x)
        return x
    

class PyramidFusionModule(nn.Module):
    def __init__(self, dim, out_channels, num_blocks):
        super(PyramidFusionModule, self).__init__()
        self.reduction_3 = nn.Conv2d(dim * 8, dim * 4, kernel_size=1, bias=False)
        self.fuse_block_3 = FuseBlocks(dim * 4, num_blocks=num_blocks[2])
        self.up_3 = Upsample(dim * 4)

        self.reduction_2 = nn.Conv2d(dim * 6, dim * 2, kernel_size=1, bias=False)
        self.fuse_block_2 = FuseBlocks(dim * 2, num_blocks=num_blocks[1])
        self.up_2 = Upsample(dim * 2)

        self.reduction_1 = nn.Conv2d(dim * 3, dim, kernel_size=1, bias=False)
        self.fuse_block_1 = FuseBlocks(dim, num_blocks=num_blocks[0])
        
        self.final_conv = nn.Conv2d(in_channels=dim, out_channels=out_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, features_a, features_b):
        x_a1, x_a2, x_a3 = features_a
        x_b1, x_b2, x_b3 = features_b
        
        x3 = self.reduction_3(torch.cat((x_a3, x_b3), dim=1))
        x3 = self.fuse_block_3(x3)
        x3_up = self.up_3(x3)

        x2 = self.reduction_2(torch.cat((x3_up, x_a2, x_b2), dim=1))
        x2 = self.fuse_block_2(x2)
        x2_up = self.up_2(x2)

        x1 = self.reduction_1(torch.cat((x2_up, x_a1, x_b1), dim=1))
        x1 = self.fuse_block_1(x1)

        output = self.final_conv(x1)
        return output
    

class SCAModulation(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SCAModulation, self).__init__()
        self.spatial_gen = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim, bias=False),
            nn.Conv2d(in_dim, out_dim * 2, kernel_size=1)
        )
        self.channel_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_dim, in_dim // 2, kernel_size=1, bias=False),
            nn.Conv2d(in_dim // 2, out_dim * 2, kernel_size=1)
        )

        nn.init.zeros_(self.spatial_gen[-1].weight)
        nn.init.zeros_(self.spatial_gen[-1].bias)
        nn.init.zeros_(self.channel_gen[-1].weight)
        nn.init.zeros_(self.channel_gen[-1].bias)

    def forward(self, guide, x):
        spatial_params = self.spatial_gen(guide)
        channel_params = self.channel_gen(guide)
        params = spatial_params + channel_params

        gamma, beta = torch.chunk(params, 2, dim=1)
        x = x * (1.0 + gamma) + beta
        return x


class Denoising_Network(nn.Module):
    def __init__(self, img_channels=1, out_channels=1, dim=64, num_blocks_a=[1, 1, 3, 1], num_blocks_b=[1, 1, 3, 1], fusion_blocks=[1, 1, 1]):
        super(Denoising_Network, self).__init__()
        # -------------------- Branch A (Source/Structure) --------------------
        self.proj_conv_a = nn.Conv2d(img_channels, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.enc_1_a = LFBlocks(dim, num_blocks=num_blocks_a[0])
        
        self.enc_down1_a = Downsample(dim)
        self.enc_2_a = LFBlocks(dim * 2, num_blocks=num_blocks_a[1])
        
        self.enc_down2_a = Downsample(dim * 2)
        self.enc_3_a = LFBlocks(dim * 4, num_blocks=num_blocks_a[2])
        
        self.enc_down3_a = Downsample(dim * 4)
        self.enc_bott_a = LFBlocks(dim * 8, num_blocks=num_blocks_a[3])

        self.dec_up3_a = Upsample(dim * 8)
        self.dec_reduce3_a = nn.Conv2d(dim * 8, dim * 4, kernel_size=1, bias=False)
        self.dec_dec3_a = LFBlocks(dim * 4, num_blocks=num_blocks_a[2])

        self.dec_up2_a = Upsample(dim * 4)
        self.dec_reduce2_a = nn.Conv2d(dim * 4, dim * 2, kernel_size=1, bias=False)
        self.dec_dec2_a = LFBlocks(dim * 2, num_blocks=num_blocks_a[1])

        self.dec_up1_a = Upsample(dim * 2)
        self.dec_reduce1_a = nn.Conv2d(dim * 2, dim * 1, kernel_size=1, bias=False)
        self.dec_dec1_a = LFBlocks(dim * 1, num_blocks=num_blocks_a[0])

        self.final_conv_a = nn.Conv2d(in_channels=dim, out_channels=out_channels, kernel_size=3, padding=1, bias=True)

        # -------------------- Branch B (Target/Texture) --------------------
        self.proj_conv_b = nn.Conv2d(img_channels, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.enc_1_b = HFBlocks(dim, num_blocks=num_blocks_b[0])

        self.enc_down1_b = Downsample(dim)
        self.enc_2_b = HFBlocks(dim * 2, num_blocks=num_blocks_b[1])

        self.enc_down2_b = Downsample(dim * 2)
        self.enc_3_b = HFBlocks(dim * 4, num_blocks=num_blocks_b[2])

        self.enc_down3_b = Downsample(dim * 4)
        self.enc_bott_b = HFBlocks(dim * 8, num_blocks=num_blocks_b[3])

        self.dec_up3_b = Upsample(dim * 8)
        self.dec_reduce3_b = nn.Conv2d(dim * 8, dim * 4, kernel_size=1, bias=False)
        self.dec_dec3_b = HFBlocks(dim * 4, num_blocks=num_blocks_b[2])

        self.dec_up2_b = Upsample(dim * 4)
        self.dec_reduce2_b = nn.Conv2d(dim * 4, dim * 2, kernel_size=1, bias=False)
        self.dec_dec2_b = HFBlocks(dim * 2, num_blocks=num_blocks_b[1])

        self.dec_up1_b = Upsample(dim * 2)
        self.dec_reduce1_b = nn.Conv2d(dim * 2, dim * 1, kernel_size=1, bias=False)
        self.dec_dec1_b = HFBlocks(dim * 1, num_blocks=num_blocks_b[0])

        self.final_conv_b = nn.Conv2d(in_channels=dim, out_channels=out_channels, kernel_size=3, padding=1, bias=True)

        # -------------------- Fusion & Modulation --------------------
        self.final_fusion = PyramidFusionModule(dim, out_channels, fusion_blocks)

        self.enc_mod_1 = SCAModulation(dim, dim)
        self.enc_mod_2 = SCAModulation(dim * 2, dim * 2)
        self.enc_mod_3 = SCAModulation(dim * 4, dim * 4)

        self.bott_mod = SCAModulation(dim * 8, dim * 8)

        self.dec_mod_3 = SCAModulation(dim * 4, dim * 4)
        self.dec_mod_2 = SCAModulation(dim * 2, dim * 2)
        self.dec_mod_1 = SCAModulation(dim, dim)

    def forward(self, inp_a, inp_b, input):
        # -------------------- Branch A --------------------
        x_a = self.proj_conv_a(inp_a)
        enc_out_1_a = self.enc_1_a(x_a)

        x_a = self.enc_down1_a(enc_out_1_a)
        enc_out_2_a = self.enc_2_a(x_a)

        x_a = self.enc_down2_a(enc_out_2_a)
        enc_out_3_a = self.enc_3_a(x_a)

        x_a = self.enc_down3_a(enc_out_3_a)
        bott_a = self.enc_bott_a(x_a)

        x_a = self.dec_up3_a(bott_a)
        x_a = self.dec_reduce3_a(torch.cat((x_a, enc_out_3_a), dim=1))
        dec_out_3_a = self.dec_dec3_a(x_a)

        x_a = self.dec_up2_a(dec_out_3_a)
        x_a = self.dec_reduce2_a(torch.cat((x_a, enc_out_2_a), dim=1))
        dec_out_2_a = self.dec_dec2_a(x_a)

        x_a = self.dec_up1_a(dec_out_2_a)
        x_a = self.dec_reduce1_a(torch.cat((x_a, enc_out_1_a), dim=1))
        dec_out_1_a = self.dec_dec1_a(x_a)

        # -------------------- Branch B --------------------
        x_b = self.proj_conv_b(inp_b)
        x_b = self.enc_mod_1(guide=enc_out_1_a, x=x_b)
        enc_out_1_b = self.enc_1_b(x_b)

        x_b = self.enc_down1_b(enc_out_1_b)
        x_b = self.enc_mod_2(guide=enc_out_2_a, x=x_b)
        enc_out_2_b = self.enc_2_b(x_b)

        x_b = self.enc_down2_b(enc_out_2_b)
        x_b = self.enc_mod_3(guide=enc_out_3_a, x=x_b)
        enc_out_3_b = self.enc_3_b(x_b)

        x_b = self.enc_down3_b(enc_out_3_b)
        x_b = self.bott_mod(guide=bott_a, x=x_b)
        bott_b = self.enc_bott_b(x_b)

        x_b = self.dec_up3_b(bott_b)
        x_b = self.dec_reduce3_b(torch.cat((x_b, enc_out_3_b), dim=1))
        x_b = self.dec_mod_3(guide=dec_out_3_a, x=x_b)
        dec_out_3_b = self.dec_dec3_b(x_b)

        x_b = self.dec_up2_b(dec_out_3_b)
        x_b = self.dec_reduce2_b(torch.cat((x_b, enc_out_2_b), dim=1))
        x_b = self.dec_mod_2(guide=dec_out_2_a, x=x_b)
        dec_out_2_b = self.dec_dec2_b(x_b)

        x_b = self.dec_up1_b(dec_out_2_b)
        x_b = self.dec_reduce1_b(torch.cat((x_b, enc_out_1_b), dim=1))
        x_b = self.dec_mod_1(guide=dec_out_1_a, x=x_b)
        dec_out_1_b = self.dec_dec1_b(x_b)
        
        # -------------------- Fusion Forward --------------------
        center_a = inp_a[:, 1:2, :, :] if inp_a.shape[1] == 3 else inp_a
        center_b = inp_b[:, 1:2, :, :] if inp_b.shape[1] == 3 else inp_b
        center = input[:, 1:2, :, :] if input.shape[1] == 3 else input

        output_a = self.final_conv_a(dec_out_1_a) + center_a
        output_b = self.final_conv_b(dec_out_1_b) + center_b

        features_a = [dec_out_1_a, dec_out_2_a, dec_out_3_a]
        features_b = [dec_out_1_b, dec_out_2_b, dec_out_3_b]
        output = self.final_fusion(features_a, features_b)

        output = output + center
        return output, output_a, output_b