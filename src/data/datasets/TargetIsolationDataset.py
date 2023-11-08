import torch.utils.data as data
import torchaudio
import os

class TargetIsolationDataset(data.Dataset):
    def __init__(self, directory='./LibriSpeech/multi_speaker', train=True):
        super(TargetIsolationDataset, self).__init__()
        if train:
            self.directory = directory + '/train/'
        else:
            self.directory = directory + '/test/'
        self.x = sorted([f for f in os.listdir(directory) if f.startswith('mixed_train_') and f.endswith('.wav')])
        self.y = sorted([f for f in os.listdir(directory) if f.startswith('source_train_') and f.endswith('.wav')])

    def __len__(self):
        return len(self.mixed_files)

    def __getitem__(self, id):
        mixed_path = os.path.join(self.directory, self.mixed_files[index])
        source_path = os.path.join(self.directory, self.source_files[index])

        mixed_waveform, _ = torchaudio.load(mixed_path)
        source_waveform, _ = torchaudio.load(source_path)
        source_txt, _ = self.directory + ".txt"

        return mixed_waveform, source_waveform, source_txt
