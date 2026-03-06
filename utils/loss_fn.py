import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import piq


class SSIMLoss(nn.Module):
    def __init__(self, data_range=1.0):
        super(SSIMLoss, self).__init__()
        self.ssim_loss = piq.SSIMLoss(data_range=data_range, reduction='mean')

    def forward(self, pred, target):
        pred = torch.clamp(pred, 0, 1.0)
        target = torch.clamp(target, 0, 1.0)
        ssim_loss = self.ssim_loss(pred, target)
        return ssim_loss


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, prediction, target):
        diff = prediction - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return torch.mean(loss)


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.eval()
        
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        
        for x in range(4):
            self.slice1.add_module(str(x), vgg[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg[x])
            
        for param in self.parameters():
            param.requires_grad = False
            
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.criterion = nn.MSELoss()

    def forward(self, pred, target):
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        x_pred1 = self.slice1(pred)
        x_target1 = self.slice1(target)
        loss1 = self.criterion(x_pred1, x_target1)

        x_pred2 = self.slice2(x_pred1)
        x_target2 = self.slice2(x_target1)
        loss2 = self.criterion(x_pred2, x_target2)
        
        return loss1 + loss2
    