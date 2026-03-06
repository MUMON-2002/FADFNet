import torch
import torch.nn as nn
from pytorch_wavelets import DTCWTForward, DTCWTInverse

class DTCWT_Transformer(nn.Module):
    def __init__(self, wave_level=3, biort='near_sym_a', qshift='qshift_a'):
        super(DTCWT_Transformer, self).__init__()
        self.wave_level = wave_level
        self.dtcwt = DTCWTForward(J=self.wave_level, biort=biort, qshift=qshift)
        self.idtcwt = DTCWTInverse(biort=biort, qshift=qshift)

    def forward(self, x):
        device = x.device
        self.dtcwt = self.dtcwt.to(device)
        self.idtcwt = self.idtcwt.to(device)

        with torch.no_grad():
            yl, yh = self.dtcwt(x)

            zero_yl = torch.zeros_like(yl).to(device)
            zero_yh = []

            for i in range(self.wave_level):
                zero_yh.append(torch.zeros_like(yh[i]).to(device))
            
            x_low = self.idtcwt((yl, zero_yh))
            x_high = self.idtcwt((zero_yl, yh))
        
        return x_low, x_high
