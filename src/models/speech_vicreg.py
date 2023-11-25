from projector import Projector
from encoder import CnnEncoder
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class speech_vicreg(nn.Module):
    def __init__(self,dim,dim2):
        '''
        dim : raw wav -> latent dim channel
        dim2 : latent dim -> project channel
        '''
        super().__init__()

        self.encoder = CnnEncoder(dim=dim,dim2=dim2)
        self.projector = Projector(dim=dim,dim2=dim2)


    def forward(self,audio):
        b,t = audio.shape 

        z = self.encoder(audio) # (b,segment,dim)
        proj = self.projector(z) # (b*segment,dim2)

        return proj



if __name__ == '__main__':
    data, samplerate = torchaudio.load('src/data/LibriSpeech/train-clean-360/14/208/14-208-0000.flac')
    model = speech_vicreg(10,20)
    output = model(data)
    print(output.size())


