import torch.utils.data as data
import torchaudio
import os

class TargetIsolationTestDataset(data.Dataset):
    def __init__(self, directory):
        super(TargetIsolationTestDataset, self).__init__()
        self.directory = directory
        self.mixed_files = sorted([f for f in os.listdir(directory) if f.startswith('mixed_eval_') and f.endswith('.wav')])
        self.text_files = sorted([f for f in os.listdir(directory) if f.startswith('mixed_eval_') and f.endswith('.txt')])

    def __len__(self):
        return len(self.mixed_files)

    def __getitem__(self, index):
        mixed_path = os.path.join(self.directory, self.mixed_files[index])
        text_path = os.path.join(self.directory, self.text_files[index])

        mixed_waveform, _ = torchaudio.load(mixed_path)
        with open(text_path, 'r') as f:
            text = f.read().strip()

        return mixed_waveform, text
