import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
import numpy as np

class Projector(nn.Module):
    def __init__(self,dim,dim2):
        super().__init__()

        sizes = [dim, dim2, dim2, dim2]

        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i],sizes[i+1],bias=False)),
            layers.append(nn.BatchNorm1d(sizes[i+1])),
            layers.append(nn.ReLU(inplace=True)),
        layers.append(nn.Linear(sizes[-2],sizes[-1],bias=False))
        self.projector = nn.Sequential(*layers)
        self.bn = nn.BatchNorm1d(sizes[-1],affine=False)

    def forward(self,x):

        b, t, d = x.shape

        z = torch.reshape(x,(b*t,-1))
        z = self.projector(z)
        #z = torch.reshape(z,(b,t,-1))

        return self.bn(z)