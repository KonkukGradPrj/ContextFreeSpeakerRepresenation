import torch.utils.data as data
import torchaudio
import os

class SepaTrain(data.Dataset):
    def __init__(self, directory=''):
        super(SepaTrain, self).__init__()
        self.directory = directory
        self.mixed_files = sorted([f for f in os.listdir(directory) if f.startswith('mixed_train_') and f.endswith('.wav')])
        self.source_files = sorted([f for f in os.listdir(directory) if f.startswith('source_train_') and f.endswith('.wav')])

    def __len__(self):
        return len(self.mixed_files)

    def __getitem__(self, index):
        mixed_path = os.path.join(self.directory, self.mixed_files[index])
        source_path = os.path.join(self.directory, self.source_files[index])

        mixed_waveform, _ = torchaudio.load(mixed_path)
        source_waveform, _ = torchaudio.load(source_path)

        return mixed_waveform, source_waveform
