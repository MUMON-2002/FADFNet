import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numbers


class XNet(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 activation="ReLU",
                 norm_type="GN"
                 ):
        super(XNet, self).__init__()

        self.levels = 4
        self.norm_type = norm_type
        self.activation = activation

        # ---------- encoder-decoder a ----------
        self.encoder_layers_a = nn.ModuleList()
        self.downconv_layers_a = nn.ModuleList()

        self.encoder_layers_a.append(ConvBlock(in_channels=inp_channels, out_channels=dim, activation=self.activation, norm_type=self.norm_type))
        for i in range(1, self.levels):
            self.downconv_layers_a.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.encoder_layers_a.append(ConvBlock(in_channels=dim * 2**(i-1), out_channels=dim * 2**i, activation=self.activation, norm_type=self.norm_type))

        self.upconv_layers_a = nn.ModuleList()
        self.decoder_layers_a = nn.ModuleList()

        for i in range(self.levels - 1, 0, -1):
            self.upconv_layers_a.append(UpConv(in_channels=dim * 2**i, out_channels=dim * 2**(i-1), activation=self.activation, norm_type=None))
            self.decoder_layers_a.append(ConvBlock(in_channels=dim * 2**i, out_channels=dim * 2**(i-1), activation=self.activation, norm_type=None))

        self.finalconv_a = nn.Conv2d(dim, out_channels, kernel_size=1)

        # ---------- encoder-decoder b ----------
        self.encoder_layers_b = nn.ModuleList()
        self.downconv_layers_b = nn.ModuleList()

        self.encoder_layers_b.append(ConvBlock(in_channels=inp_channels, out_channels=dim, activation=self.activation, norm_type=self.norm_type))
        for i in range(1, self.levels):
            self.downconv_layers_b.append(nn.MaxPool2d(kernel_size=2, stride=2)),
            self.encoder_layers_b.append(ConvBlock(in_channels=dim * 2**(i-1), out_channels=dim * 2**i, activation=self.activation, norm_type=self.norm_type))

        self.upconv_layers_b = nn.ModuleList()
        self.decoder_layers_b = nn.ModuleList()

        for i in range(self.levels - 1, 0, -1):
            self.upconv_layers_b.append(UpConv(in_channels=dim * 2**i, out_channels=dim * 2**(i-1), activation=self.activation, norm_type=None))
            self.decoder_layers_b.append(ConvBlock(in_channels=dim * 2**i, out_channels=dim * 2**(i-1), activation=self.activation, norm_type=None))

        self.finalconv_b = nn.Conv2d(dim, out_channels, kernel_size=1)

        # ---------- intermidiate fusion ----------
        self.inter_fusion = OTG_CrossAttentionFusion(in_channels=dim * 2 ** (self.levels - 1), norm_type=None)

        # ---------- output fusion ----------
        self.out_fusion = FeaturePyramidFusion(inp_channels=1, dim=dim)

    def forward(self, inp_a, inp_b):

        # ---------- encoder ----------
        x_a = inp_a 
        enc_outputs_a = []
        for i, encoder_layer in enumerate(self.encoder_layers_a): 
            x_a = encoder_layer(x_a)
            enc_outputs_a.append(x_a)
            if i < self.levels - 1: 
                x_a = self.downconv_layers_a[i](x_a) 

        x_b = inp_b 
        enc_outputs_b = []
        for i, encoder_layer in enumerate(self.encoder_layers_b): 
            x_b = encoder_layer(x_b)
            enc_outputs_b.append(x_b)
            if i < self.levels - 1: 
                x_b = self.downconv_layers_b[i](x_b) 

        # ---------- inter fusion ----------
        x_a, x_b = self.inter_fusion(x_a, x_b)

        # ---------- decoder ----------
        for i in range(self.levels - 1): 
            x_a = self.upconv_layers_a[i](x_a) 
            x_a = torch.cat((x_a, enc_outputs_a[-(i+2)]), dim=1) 
            x_a = self.decoder_layers_a[i](x_a)

        output_a = self.finalconv_a(x_a)

        output_a = output_a + inp_a
                
        for i in range(self.levels - 1): 
            x_b = self.upconv_layers_b[i](x_b) 
            x_b = torch.cat((x_b, enc_outputs_b[-(i+2)]), dim=1) 
            x_b = self.decoder_layers_b[i](x_b)

        output_b = self.finalconv_b(x_b)
        output_b = output_b + inp_b

        # ---------- out fusion ----------
        output = self.out_fusion(output_a, output_b)

        return output


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, activation="ReLU", norm_type=None):
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = get_norm_layer(norm_type, out_channels)
        self.act1 = get_activation(activation)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = get_norm_layer(norm_type, out_channels)
        self.act2 = get_activation(activation)
    
    def forward(self, x):

        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))

        return x


class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels, activation="ReLU", norm_type=None):
        super(UpConv, self).__init__()
        
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.norm = get_norm_layer(norm_type, out_channels)
        self.act = get_activation(activation)
    
    def forward(self, x):

        return self.act(self.norm(self.upconv(x)))


class OTG_CrossAttentionFusion(nn.Module):

    def __init__(self, in_channels, norm_type=None):
        super(OTG_CrossAttentionFusion, self).__init__()

        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))

        self.ot = OptimalTransport()

        self.norm_a = get_norm_layer(norm_type, in_channels)
        self.norm_b = get_norm_layer(norm_type, in_channels)
        
        self.qkv_A = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 3, kernel_size=1, bias=False),
            nn.Conv2d(in_channels * 3, in_channels * 3, kernel_size=3, stride=1, padding=1, groups=in_channels * 3, bias=False)
        )
        self.out_A = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.qkv_B = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 3, kernel_size=1, bias=False),
            nn.Conv2d(in_channels * 3, in_channels * 3, kernel_size=3, stride=1, padding=1, groups=in_channels * 3, bias=False)
        )
        self.out_B = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, inp_a, inp_b):
        
        B, C, H, W = inp_a.shape
        identity_a = inp_a
        identity_b = inp_b

        inp_a = self.norm_a(inp_a)
        inp_b = self.norm_b(inp_b)

        Trans_a = self.ot(inp_a, inp_b).view(B, H, W, H, W) # (B, H_b, W_b, H_a, W_a)
        Trans_b = self.ot(inp_b, inp_a).view(B, H, W, H, W) # (B, H_a, W_a, H_b, W_b)

        qkv_a = self.qkv_A(inp_a).view(B, C * 3, H, W)
        query_A, key_A, value_A = qkv_a.chunk(3, dim=1)
        qkv_b = self.qkv_B(inp_b).view(B, C * 3, H, W)
        query_B, key_B, value_B = qkv_b.chunk(3, dim=1)
        
        attn_A = torch.einsum("bchw, bcyx -> bhwyx", query_B, key_A).contiguous() / math.sqrt(C) # (B, H_b, W_b, H_a, W_a)
        attn_A = attn_A + self.alpha * Trans_a
        attn_A = torch.softmax(attn_A.view(B, H, W, -1), -1).view(B, H, W, H, W)
        out_a = torch.einsum("bhwyx, bcyx -> bchw", attn_A, value_A).contiguous()
        out_a = self.out_A(out_a) + identity_a

        attn_B = torch.einsum("bchw, bcyx -> bhwyx", query_A, key_B).contiguous() / math.sqrt(C) # (B, H_a, W_a, H_b, W_b)
        attn_B = attn_B + self.beta * Trans_b
        attn_B = torch.softmax(attn_B.view(B, H, W, -1), -1).view(B, H, W, H, W)
        out_b = torch.einsum("bhwyx, bcyx -> bchw", attn_B, value_B).contiguous()
        out_b = self.out_B(out_b) + identity_b

        return out_a, out_b
    

class OptimalTransport(nn.Module):

    def __init__(self, reg=1e-1, max_iter=50, transport_ratio=0.8):
        super().__init__()
        self.reg = reg
        self.max_iter = max_iter
        self.transport_ratio = transport_ratio

    def sinkhorn(self, a, b, cost_matrix, eps=1e-6):

        K = torch.exp(-cost_matrix / self.reg).clamp(min=1e-8)

        u = torch.ones_like(a)
        v = torch.ones_like(b)

        for _ in range(self.max_iter):
            prev_u, prev_v = u.clone(), v.clone()
            u = a / (torch.bmm(K, v.unsqueeze(2)).squeeze(2) + eps)
            v = b / (torch.bmm(K.transpose(1, 2), u.unsqueeze(2)).squeeze(2) + eps)

            err_u = torch.max(torch.abs(u - prev_u)).item()
            err_v = torch.max(torch.abs(v - prev_v)).item()
            if err_u < eps and err_v < eps:
                break

        transport_plan = u.unsqueeze(2) * K * v.unsqueeze(1)
        return transport_plan

    def forward(self, source, target):

        B, C, H, W = source.shape
        N = H * W
        device = source.device
        src = source.view(B, C, -1).permute(0, 2, 1) # (B, N, C)
        tgt = target.view(B, C, -1).permute(0, 2, 1)

        src_weights = torch.ones(B, N, device=device) * (self.transport_ratio / N)
        tgt_weights = torch.ones(B, N, device=device) * (self.transport_ratio / N)
        virtual_src_weight = torch.ones(B, 1, device=device) * (1 - self.transport_ratio)
        virtual_tgt_weight = torch.ones(B, 1, device=device) * (1 - self.transport_ratio)
        partial_src_weights = torch.cat([src_weights, virtual_src_weight], dim=1)
        partial_tgt_weights = torch.cat([tgt_weights, virtual_tgt_weight], dim=1)

        cost = torch.cdist(src, tgt, p=2) ** 2
        virtual_cost = torch.quantile(cost.view(B, -1), q=0.75, dim=1)[0].unsqueeze(-1).unsqueeze(-1)

        partial_cost = torch.zeros(B, N + 1, N + 1, device=device)
        partial_cost[:, :N, :N] = cost
        partial_cost[:, :N, N:] = virtual_cost.expand(B, N, 1)
        partial_cost[:, N:, :N] = virtual_cost.expand(B, 1, N)

        transport_plan = self.sinkhorn(partial_src_weights, partial_tgt_weights, partial_cost) 
        
        transport_plan = transport_plan[:, :N, :N]
        transport_plan = transport_plan.permute(0, 2, 1)
        transport_plan = transport_plan / transport_plan.sum(dim=2, keepdim=True)
        
        return transport_plan
    

class Embed_Fusion_3_3(nn.Module):

    def __init__(self, dim):
        super(Embed_Fusion_3_3, self).__init__()

        self.conv = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1)

    def forward(self, inp_a, inp_b):
        x = torch.concat([inp_a, inp_b], dim=1)
        x = self.conv(x)
        return x


class Embed_Fusion_1_1(nn.Module):

    def __init__(self, dim):
        super(Embed_Fusion_1_1, self).__init__()

        self.conv = nn.Conv2d(dim * 2, dim, kernel_size=1, stride=1, bias=False)

    def forward(self, inp_a, inp_b):
        x = torch.concat([inp_a, inp_b], dim=1)
        x = self.conv(x)
        return x


class FeaturePyramidFusion(nn.Module): 
    
    def __init__(self, inp_channels, dim, activation="ReLU"): 
        super(FeaturePyramidFusion, self).__init__() 

        self.down_layer_1_A = nn.Sequential(
            nn.Conv2d(inp_channels, dim, kernel_size=3, padding=1),
            get_activation(activation),
        )
        self.down_layer_2_A = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=3, padding=1),
            get_activation(activation),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.down_layer_3_A = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 4, kernel_size=3, padding=1),
            get_activation(activation),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.down_layer_1_B = nn.Sequential(
            nn.Conv2d(inp_channels, dim, kernel_size=3, padding=1),
            get_activation(activation),
        )
        self.down_layer_2_B = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=3, padding=1),
            get_activation(activation),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.down_layer_3_B = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 4, kernel_size=3, padding=1),
            get_activation(activation),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.up_layer_3 = nn.Sequential(
            nn.Conv2d(dim * 4, dim * 2 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            get_activation(activation),
        )
        self.up_layer_2 = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 1 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            get_activation(activation),
        )
        self.up_layer_1 = nn.Sequential(
            nn.Conv2d(dim * 1, inp_channels, kernel_size=3, stride=1, padding=1),
            get_activation(activation),
        )

        self.add_fusion_1 = Embed_Fusion_1_1(dim)
        self.add_fusion_2 = Embed_Fusion_1_1(dim * 2)
        self.add_fusion_3 = Embed_Fusion_1_1(dim * 4)

        self.embed_fusion_2 = Embed_Fusion_3_3(dim * 2)
        self.embed_fusion_1 = Embed_Fusion_3_3(dim)
        
    def forward(self, inp_a, inp_b):
        c0 = inp_a + inp_b

        c1_a = self.down_layer_1_A(inp_a)
        c1_b = self.down_layer_1_B(inp_b)

        c1 = self.add_fusion_1(c1_a, c1_b)

        c2_a = self.down_layer_2_A(c1_a)
        c2_b = self.down_layer_2_B(c1_b)

        c2 = self.add_fusion_2(c2_a, c2_b)

        c3_a = self.down_layer_3_A(c2_a)
        c3_b = self.down_layer_3_B(c2_b)

        c3 = self.add_fusion_3(c3_a, c3_b)
        
        # Pyramid Fusion
        c3 = self.up_layer_3(c3)

        c2 = self.embed_fusion_2(c2, c3)
        c2 = self.up_layer_2(c2)
        
        c1 = self.embed_fusion_1(c1, c2)
        c1 = self.up_layer_1(c1)

        output = c0 + c1 # residual connect
        return output


def get_norm_layer(norm_type, dim):

    assert norm_type in ["BN", "GN", "IN", None]

    if norm_type == "BN":
        return nn.BatchNorm2d(dim)
    elif norm_type == "GN":
        return nn.GroupNorm(num_groups=8, num_channels=dim)
    elif norm_type == "IN":
        return nn.InstanceNorm2d(dim)
    else:
        return nn.Identity()


def get_activation(activation, negative_slope=0.1):

    assert activation in ["ReLU", "GeLU", "LeakyReLU", None]

    if activation == "ReLU":
        return nn.ReLU(inplace=True)
    elif activation == "GeLU":
        return nn.GELU()
    elif activation == "LeakyReLU":
        return nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
    else:
        return nn.Identity()
