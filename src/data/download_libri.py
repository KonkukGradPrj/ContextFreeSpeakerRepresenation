import torchaudio

# Download the LIBRISPEECH dataset
# train_clean_360 = torchaudio.datasets.LIBRISPEECH("./", url="train-clean-360", download=True)
torchaudio.datasets.LIBRISPEECH("./LibriSpeech", url="test-clean", download=True)