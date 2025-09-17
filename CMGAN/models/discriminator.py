import numpy as np
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, in_channel = 2, channels=16):
        super().__init__()

        self.network = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(in_channel, channels, (4,4), (2,2), (1,1), bias=False)
            ),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),

            nn.utils.spectral_norm(
                nn.Conv2d(channels, 2*channels, (4,4), (2,2), (1,1), bias=False)
            ),
            nn.InstanceNorm2d(2*channels, affine=True),
            nn.PReLU(2*channels),

            nn.utils.spectral_norm(
                nn.Conv2d(2*channels, 4*channels, (4,4), (2,2), (1,1), bias=False)
            ),
            nn.InstanceNorm2d(4*channels, affine=True),
            nn.PReLU(4*channels),

            nn.utils.spectral_norm(
                nn.Conv2d(4*channels, 8*channels, (4,4), (2,2), (1,1), bias=False)
            ),
            nn.InstanceNorm2d(8*channels, affine=True),
            nn.PReLU(8*channels),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(8*channels, 4*channels)),
            nn.Dropout(0.3),
            nn.PReLU(4*channels),
            nn.utils.spectral_norm(nn.Linear(4*channels,1)),
            LearnableSigmoid(1)
        )
    
    def forward(self, x, y):
        Q_pesq = self.network(torch.cat([x,y], dim=1))
        return Q_pesq
    

class LearnableSigmoid(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


