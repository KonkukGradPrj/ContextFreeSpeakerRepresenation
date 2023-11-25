import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
import numpy as np

# how to encode? CNN? Titanet?

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class Conv1dLayer(nn.Module):
    def __init__(self,in_dim,out_dim,kernel_size,stride,padding=0,bias=False):
        super().__init__()

        self.conv = nn.Conv1d(in_dim,out_dim,kernel_size=kernel_size,stride=stride,padding=padding,bias=bias)
        self.bn = nn.BatchNorm1d(out_dim)
        self.act = nn.LeakyReLU()

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
    
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

    def forward(self,z):

        b, t, d = z.shape

        #z = torch.reshape(z,(b*t,-1))
        z = self.projector(z)
        z = self.bn(z)

        #return torch.reshape(z,(b,t,-1))

class CnnEncoder(nn.Module):
    def __init__(self,dim, dim2):
        super().__init__()

        self.enc = nn.Sequential(
            Conv1dLayer(1,dim,kernel_size=10,stride=5,padding=0,bias=False),
            Conv1dLayer(dim,dim,kernel_size=10,stride=5,padding=0,bias=False),
            Conv1dLayer(dim,dim,kernel_size=10,stride=5,padding=0,bias=False),
            Conv1dLayer(dim,dim,kernel_size=8,stride=4,padding=0,bias=False),
            Conv1dLayer(dim,dim,kernel_size=4,stride=2,padding=0,bias=False),
            Conv1dLayer(dim,dim,kernel_size=4,stride=2,padding=0,bias=False),
            nn.Conv1d(dim,dim,kernel_size=4,stride=2,padding=0,bias=False),
            LambdaLayer(lambda x: x.transpose(1,2)) # 각 segment마다 dim개의 feature를 갖도록 (b segment feature)
        )
        self.project = Projector(dim,dim2)

        

    def forward(self, x):
        x = x.unsqueeze(1)
        z = self.enc(x)
        #prj = self.project(z)
    
        return z

if __name__ == '__main__':
    
    enc = CnnEncoder(128,512)

    data, samplerate = sf.read('src/data/LibriSpeech/train-clean-360/14/208/14-208-0000.flac')
    data = torch.Tensor(data)
    data = data.unsqueeze(0)
    
    
    output = enc(data)
    print(output.size())
    
    