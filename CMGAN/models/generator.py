from models.module_blocks import *

import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_channel, channels=64):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, channels, (1,1), (1,1)),     #in_channel為3，{mag,real,imag}
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels)
        )
        self.dilated_densenet = Dilated_DenseNet(in_channels = channels) 
        self.conv_2 = nn.Sequential(
            nn.Conv2d(channels, channels, (1,3), (1,2), padding=(0,1)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels)
        )
    def forward(self,x):     
        out = self.conv_1(x)
        out = self.dilated_densenet(out)
        out= self.conv_2(out)
        return out

class TS_Conformer(nn.Module):
    def __init__(self, channels=64):
        super().__init__()     
        self.time_conformer = Conformer(dim=channels)
        self.freq_conformer = Conformer(dim=channels)
    def forward(self, x):
        #Reshape 1
        B, C, T, F = x.size()
        x_in = x.permute(0,3,2,1).contiguous().view(B*F,T,C)
        #Conformer-Time 
        x_t = self.time_conformer(x_in) + x_in  #Residual 1
        #Reshape 2
        x_t = x_t.contiguous().view(B*T,F,C) 
        #Conformer-Freq
        x_f = self.freq_conformer(x_t) + x_t    #Residual 2
        #Reshape 3
        x_out = x_t.contiguous().view(B,T,F,C).permute(0,3,1,2)
        return x_out

class Mask_Decoder(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.dilated_conv = Dilated_DenseNet(in_channels = channels) 
        self.subpixel = SP_conv(in_channel = channels, out_channel = channels, r = 2)

        self.conv1 = nn.Sequential(
         #透過kernel size(1,2) to make (4,64,321,202) -> (4,1,321,201)
            nn.Conv2d(in_channels = channels, out_channels = 1, kernel_size=(1, 2)),  
            nn.InstanceNorm2d(1, affine=True),
            nn.PReLU(1)
        )
        self.conv2 = nn.Conv2d(1, 1, kernel_size = (1,1))
        self.prelu_out = nn.PReLU(201, init = -0.25)     #如論文中是對著frequency維度做activation
    def forward(self, x):
        out = self.dilated_conv(x)
        out = self.subpixel(out)
        out = self.conv1(out)
        out = self.conv2(out).permute(0,3,2,1)
        out = self.prelu_out(out).permute(0,3,2,1)
        return out

class Complex_Decoder(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.dilated_conv = Dilated_DenseNet(in_channels = channels) 
        self.subpixel = SP_conv(in_channel = channels, out_channel = channels, r = 2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = channels, out_channels = 2, kernel_size=(1,2)),
            nn.InstanceNorm2d(2, affine=True),
            nn.PReLU(2)
        )
        self.conv2 = nn.Conv2d(2, 2, kernel_size = (1,1))
    def forward(self, x):
        out = self.dilated_conv(x)
        out = self.subpixel(out)
        out = self.conv1(out)
        out = self.conv2(out)
        return out

#-----------------------------------------------------------------------------------------------------------------------------#
#以上模塊組合在一起
class Generator(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.encoder = Encoder(in_channel =3, channels=64)

        self.TS_conformer1 = TS_Conformer(channels=channels)
        #self.TS_conformer2 = TS_Conformer(channels=channels)
        #self.TS_conformer3 = TS_Conformer(channels=channels)
        #self.TS_conformer4 = TS_Conformer(channels=channels)

        self.mask_decoder = Mask_Decoder(channels = channels)
        self.complex_decoder = Complex_Decoder(channels = channels)

    def forward(self, x):
        #Encoder
        n_mag = x[:,0,:,:].unsqueeze(1)
        n_phase = torch.angle(torch.complex(x[:,1,:,:], x[:,2,:,:])).unsqueeze(1)
        x = self.encoder(x)
        #Two-stage conformer * 4
        x = self.TS_conformer1(x)
        #x = self.TS_conformer2(x)
        #x = self.TS_conformer3(x)
        #x = self.TS_conformer4(x)
        #Decoder
        mask = self.mask_decoder(x)
        out_mag = mask * n_mag
        complex = self.complex_decoder(x)
        #Output
        out_real = out_mag*torch.cos(n_phase) + complex[:,0,:,:].unsqueeze(1)
        out_imag = out_mag*torch.sin(n_phase) + complex[:,1,:,:].unsqueeze(1)
        return out_real, out_imag