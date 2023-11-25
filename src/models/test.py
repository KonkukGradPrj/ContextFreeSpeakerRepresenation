import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from glob import glob
import soundfile as sf
import matplotlib.pyplot as plt
import librosa

data, samplerate = sf.read('src/data/LibriSpeech/train-clean-360/14/208/14-208-0000.flac')



if __name__ == '__main__':
    print(len(data))
    print(data)
    
