import soundfile as sf
import librosa
import torch
import torchaudio
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader,Dataset
from glob import glob

def find_audio_path(is_train,is_multi,root = '/home/hyeons/workspace/ContextFreeSpeakerRepresenation/src/data/LibriSpeech'):
    
    final_path = []
    if is_train:
        if is_multi:
            path = glob(root+'/train-multi/*')
        else:
            path = glob(root+'/train-clean-360/*')
    else:
        if is_multi:
            path = glob(root+'/test-multi/*')
        else:
            path = glob(root+'/test-clean/*')

    for path_ in path:
        tmp = glob(path_+'/*')
        final_path.extend(tmp)

    return final_path


class AudioDataset(Dataset):
    
    def __init__(self,path):

        self.path = path
        super().__init__()

    def __len__(self):
        return len(self.path)

    def __getitem__(self, index):

        result = dict()
        tmp_path = glob(self.path[index]+'/*.flac')
        label_path = glob(self.path[index]+'/*.txt')
        selected_file = np.random.choice(tmp_path,1)

        audio,sr = torchaudio.load(selected_file[0])
        

        result['audio'] = audio
        file_num = int(selected_file[0].split('/')[-1].split('.')[0].split('-')[-1])

        with open(label_path[0]) as f:
            label = f.readlines()[file_num]

        result['label'] = label[17:]

        return result

class AudioTestDataset(Dataset):
    
    def __init__(self,path):

        self.path = path
        super().__init__()

    def __len__(self):
        return len(self.path)

    def __getitem__(self, index):

        result = dict()
        tmp_path = glob(self.path[index]+'/*.flac')
        label_path = glob(self.path[index]+'/*.txt')
        selected_file = np.random.choice(tmp_path,1)

        audio,sr = torchaudio.load(selected_file[0])
        

        result['audio'] = audio
        file_num = int(selected_file[0].split('/')[-1].split('.')[0].split('-')[-1])

        with open(label_path[0]) as f:
            label = f.readlines()[file_num]

        result['label'] = label[17:]

        return result
        



if __name__ == '__main__':
    is_train = True
    is_multi = False
    
    path = find_audio_path(is_train=is_train,is_multi=is_multi)

    dataset = AudioDataset(path)
    ######### Loader tensor 크기 다른거 수정만 해주면됌 ##################
    loader = DataLoader(dataset,batch_size=2)
    
    
    