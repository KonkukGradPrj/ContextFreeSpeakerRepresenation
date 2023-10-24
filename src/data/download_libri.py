import torchaudio

# Download the LIBRISPEECH dataset
train_clean_360 = torchaudio.datasets.LIBRISPEECH("./clean", url="train-clean-360", download=True)