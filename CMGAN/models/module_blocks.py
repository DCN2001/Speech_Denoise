import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

class Dilated_DenseNet(nn.Module):
    def __init__(self, in_channels=64):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = (2, 3)  #按照原版設定
        
        #d = 1
        self.d1 = nn.Sequential(
            #對input做padding目的: 確保輸入輸出同一個size(受kernel size影響)，而kernel size經過dilation變大，就必須調整padding的量
            nn.ConstantPad2d((1,1,1,0), value=0.0),      
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=self.kernel_size, dilation=(1,1)),  #dilated目的為增加Receptive field(只對time軸dilate)
            nn.InstanceNorm2d(self.in_channels, affine=True),
            nn.PReLU(self.in_channels)
        )
        
        #d = 2
        self.d2 = nn.Sequential(
            #nn.ConstantPad2d((1,1,1,1), value=0.0),  #使用conv2d內建則是兩邊都會padding
            nn.Conv2d(self.in_channels*2, self.in_channels, kernel_size=self.kernel_size, padding=(1,1), dilation=(2,1)),
            nn.InstanceNorm2d(self.in_channels, affine=True),
            nn.PReLU(self.in_channels)
        )

        #d = 4
        self.d4 = nn.Sequential(
            #nn.ConstantPad2d((1,1,2,2), value=0.0),   #使用conv2d內建則是兩邊都會padding
            nn.Conv2d(self.in_channels*3, self.in_channels, kernel_size=self.kernel_size, padding=(2,1), dilation=(4,1)),
            nn.InstanceNorm2d(self.in_channels, affine=True),
            nn.PReLU(self.in_channels)
        )

        #d = 8
        self.d8 = nn.Sequential(
            #nn.ConstantPad2d((1,1,4,4), value=0.0),   #使用conv2d內建則是兩邊都會padding
            nn.Conv2d(self.in_channels*4, self.in_channels, kernel_size=self.kernel_size, padding=(4,1), dilation=(8,1)),
            nn.InstanceNorm2d(self.in_channels, affine=True),
            nn.PReLU(self.in_channels)
        )

    def forward(self,x):
        skip = x
        out = self.d1(skip)
        skip1 = torch.cat([out,skip], dim=1)
        out = self.d2(skip1)
        skip2 = torch.cat([out,skip1], dim=1)
        out = self.d4(skip2)
        skip3 = torch.cat([out, skip2], dim=1)
        out = self.d8(skip3)
        return out
     
#---------------------------------------------------------------------------------------------------------------------------------
#Sub-pixel
class SP_conv(nn.Module):
    def __init__(self, in_channel, out_channel, r):
        super().__init__()
        self.kernel_size = (1,3)
        self.r = r
        self.conv = nn.Conv2d(in_channel, out_channel*r, kernel_size=self.kernel_size, stride=(1,1), padding=(0,1))

    def forward(self,x):
        out = self.conv(x)
        #要轉變tensor形狀
        B, C, T, F = out.size()
        out = torch.reshape(out, (B,C//self.r,T,F*self.r))
        return out

#-----------------------------------------------------------------------------------------------------------------------------------
#Conformer Block
class Conformer(nn.Module):
    def __init__(self, dim, heads=4 ,ff_mult=4, conv_expansion=2, conv_kernel_size=31, conv_padding=(15,15), dropout=0.0):
        super().__init__()
        self.prenorm1 = nn.LayerNorm(dim)
        self.FFNN1 = nn.Sequential(
            nn.Linear(dim, dim*ff_mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim*ff_mult, dim),
            nn.Dropout(dropout)
        )

        self.prenorm2 = nn.LayerNorm(dim)
        self.MSHA = MHSA(dim, heads, dim_head=64, dropout=dropout)

        inner_dim = dim*conv_expansion
        self.Conv_module = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n c -> b c n'),     #reshape for conv1d on dim of n
            nn.Conv1d(dim, inner_dim*2, 1),     # reason of inner_dim*2:  for the next GLU because it will split the dim to half
            nn.GLU(dim=1),
            Depthwise_conv(inner_dim, inner_dim, conv_kernel_size, conv_padding),
            nn.BatchNorm1d(inner_dim),
            nn.Conv1d(inner_dim, dim, 1),
            Rearrange('b c n -> b n c'),
            nn.Dropout(dropout) 
        )

        self.prenorm3 = nn.LayerNorm(dim)
        self.FFNN2 = nn.Sequential(
            nn.Linear(dim, dim*ff_mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim*ff_mult, dim),
            nn.Dropout(dropout)
        )

        self.postnorm = nn.LayerNorm(dim)

    def forward(self, x_in):
        x = self.prenorm1(x_in) 
        x_f1 = self.FFNN1(x) *0.5 + x_in    #Residual 1
        x = self.prenorm2(x_f1) 
        x_attn = self.MSHA(x) + x_f1        #Residual 2
        x_conv = self.Conv_module(x_attn) + x_attn
        x = self.prenorm3(x_conv)
        x = self.FFNN2(x) * 0.5 + x_conv    #Residual 3
        x = self.postnorm(x)
        return x


#----------------------------------------------------------------------------------------------------------------------
#Conformer Block 小組件
class MHSA(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = heads *  dim_head 
        self.query = nn.Linear(dim, inner_dim, bias=False)
        self.key = nn.Linear(dim, inner_dim, bias=False)
        self.value = nn.Linear(dim, inner_dim, bias=False)
        self.m_attn = nn.MultiheadAttention(embed_dim=inner_dim, num_heads=heads)
        self.ff_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        out, attn = self.m_attn(q,k,v, need_weights=False)
        out = self.ff_out(out)
        return self.dropout(out)

class Depthwise_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, groups=in_channels)
    def forward(self, x):
        x = F.pad(x, self.padding)
        out = self.conv(x)
        return out