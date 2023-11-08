import os
import torch
import torchaudio
import random

def adjust_noise_length(noise_wave, target_length):
    # Adjust the length of noise_wave to match the target_length
    if noise_wave.shape[1] > target_length:
        # If noise is longer than source, crop noise
        noise_wave = noise_wave[:, :target_length]
    elif noise_wave.shape[1] < target_length:
        # If noise is shorter than source, pad noise
        padding = target_length - noise_wave.shape[1]
        noise_wave = torch.nn.functional.pad(noise_wave, (0, padding))
    return noise_wave

def add_noise(dataset, source_wave, source_id):
    mixed_wave = source_wave.clone()
    noise_rate = 0.0
    rate_lst = [0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.5]
    source_length = source_wave.shape[1]  # Get the length of the source_wave once

    while noise_rate < 1.0:
        index = random.randint(0, len(dataset) - 1)
        noise_wave, _, _, noise_id, _, _ = dataset[index]

        if noise_id != source_id:
            noise_wave = adjust_noise_length(noise_wave, source_length)
            rate = random.choice(rate_lst)
            mixed_wave += noise_wave * rate
            noise_rate += rate

    return mixed_wave

def make_trainset():
    # Initialize the dataset
    dataset_path = os.path.join(".")
    librispeech_dataset = torchaudio.datasets.LIBRISPEECH("./", url="train-clean-360", download=False)

    # Create a directory for training samples
    train_output_dir = os.path.join(".", "LibriSpeech", "train-multi")
    os.makedirs(train_output_dir, exist_ok=True)

    source_id_counts = {}  # Dictionary to track count for each source_id
    print("start making train set...")
    for idx in range(len(librispeech_dataset)):  
        source_wave, sample_rate, source_script, source_id, _, _ = librispeech_dataset[idx]

        mixed_wave = add_noise(librispeech_dataset, source_wave, source_id)

        # Check if source_id is already in the dictionary
        if source_id not in source_id_counts:
            source_id_counts[source_id] = 1
        else:
            source_id_counts[source_id] += 1

        # Save mixed_wave, source_wave, and source_script
        data_output_dir = os.path.join(train_output_dir, f"{source_id}")
        os.makedirs(data_output_dir, exist_ok=True)
        
        # Generate unique filenames using source_id and its count from the dictionary
        filename_prefix = os.path.join(data_output_dir, f"{source_id}_{source_id_counts[source_id]}")
        
        torchaudio.save(f"{filename_prefix}_mixed.flac", mixed_wave, sample_rate)
        torchaudio.save(f"{filename_prefix}_source.flac", source_wave, sample_rate)
        
        # Save the script to a text file
        with open(f"{filename_prefix}_script.txt", 'w') as f:
            f.write(source_script)



def make_testset():
    # Initialize the dataset
    dataset_path = os.path.join(".")
    librispeech_dataset = torchaudio.datasets.LIBRISPEECH("./", url="test-clean", download=False)

    # Create a directory for training samples
    test_output_dir = os.path.join(".", "LibriSpeech", "test-multi")
    os.makedirs(test_output_dir, exist_ok=True)

    source_id_counts = {}  # Dictionary to track count for each source_id
    print("start making test set...")
    for idx in range(len(librispeech_dataset)):  
        source_wave, sample_rate, source_script, source_id, _, _ = librispeech_dataset[idx]

        mixed_wave = add_noise(librispeech_dataset, source_wave, source_id)

        # Check if source_id is already in the dictionary
        if source_id not in source_id_counts:
            source_id_counts[source_id] = 1
        else:
            source_id_counts[source_id] += 1

        # Save mixed_wave, source_wave, and source_script
        data_output_dir = os.path.join(test_output_dir, f"{source_id}")
        os.makedirs(data_output_dir, exist_ok=True)
        
        # Generate unique filenames using source_id and its count from the dictionary
        filename_prefix = os.path.join(data_output_dir, f"{source_id}_{source_id_counts[source_id]}")
        
        torchaudio.save(f"{filename_prefix}_mixed.flac", mixed_wave, sample_rate)
        torchaudio.save(f"{filename_prefix}_source.flac", source_wave, sample_rate)
        
        # Save the script to a text file
        with open(f"{filename_prefix}_script.txt", 'w') as f:
            f.write(source_script)

if __name__ =="__main__":
    make_trainset()
    make_testset()