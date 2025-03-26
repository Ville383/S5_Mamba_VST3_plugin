from torch.utils.data import Dataset
import os
import torch
import torchaudio
from tqdm import tqdm

class AudioDataset(Dataset):
    def __init__(self, input, target, sequence_length, truncate_length):
        self.input_sequence = self.wrap_to_sequences(input, sequence_length, truncate_length)
        self.target_sequence = self.wrap_to_sequences(target, sequence_length, truncate_length)
        self.length = len(self.input_sequence)

    def __getitem__(self, index):
        # Transpose from (channels, length) -> (length, channels)
        return self.input_sequence[index].T, self.target_sequence[index].T

    def __len__(self):
        return self.length

    def wrap_to_sequences(self, waveform, sequence_length, overlap):
        assert waveform.size(1) >= sequence_length, "The length of audio must be equal or higher than sequence length"
        step = sequence_length - overlap
        return [waveform[:, i:i + sequence_length] for i in range(0, waveform.size(1) - sequence_length + 1, step)]
    

def parse_conditioning(filename):
    """Extracts (d, t) values from filename and scales them to [-1, 1]."""
    parts = filename.split("_")
    d = int(parts[1][1])  # Extract d value [0, 4]
    t = int(parts[2][1])  # Extract t value [0, 4]
    d_scaled = 2 * (d / 4) - 1  # Scale to [-1, 1]
    t_scaled = 2 * (t / 4) - 1  # Scale to [-1, 1]
    return torch.tensor([d_scaled, t_scaled], dtype=torch.float32)


def createDataset(args):
    data_dir = os.path.join("boss_od3_overdrive", "overdrive", "boss_od3")
    x_dir = os.path.join(data_dir, "x")
    y_dir = os.path.join(data_dir, "y")

    # Get list of x and y files
    x_files = {f[1:]: os.path.join(x_dir, f) for f in os.listdir(x_dir) if f.endswith(".wav")}  # Remove 'x' prefix
    y_files = {f[1:]: os.path.join(y_dir, f) for f in os.listdir(y_dir) if f.endswith(".wav")}  # Remove 'y' prefix

    train_data = []
    val_data = []
    test_data = []

    def split_audio(x_path, y_path, c, seq_len):
        """Splits the audio files into `seq_len` equal parts while keeping `c` the same."""

        # Load the audio from the file paths
        x, _ = torchaudio.load(x_path, channels_first=False)  # (samples, 1)
        y, _ = torchaudio.load(y_path, channels_first=False)  # (samples, 1), sometimes y has two channels (take into account when appending to data)

        # Ensure both have the same length
        x_len = x.shape[0]
        y = y[:x_len]

        # Divide to train, val, and test sets 80/15/5
        x_train = x[:int(0.8*x_len)]
        y_train = y[:int(0.8*x_len)]

        x_val = x[int(0.8*x_len):int(0.95*x_len)]
        y_val = y[int(0.8*x_len):int(0.95*x_len)]

        x_test = x[int(0.95*x_len):]
        y_test = y[int(0.95*x_len):]    
    
        # Calculate number of splits
        x_train_len = x_train.shape[0]
        assert seq_len <= x_train_len, "sequence_length must be less or the same length as training data"
        num_splits = x_train_len // seq_len  # Integer division

        # Create multiple segments
        for i in range(num_splits):
            start = i * seq_len
            end = start + seq_len
            train_data.append({
                "input": x_train[start:end, 0].unsqueeze(-1),  # (split_size, 1)
                "target": y_train[start:end, 0].unsqueeze(-1),  # (split_size, 1)
                "c": c  # Keep conditioning unchanged
            })

        x_val_len = x_val.shape[0]
        assert seq_len <= x_train_len, "sequence_length must be less or the same length as validation data"
        num_splits = x_val_len // seq_len  # Integer division

        # Create multiple segments
        for i in range(num_splits):
            start = i * seq_len
            end = start + seq_len
            val_data.append({
                "input": x_val[start:end, 0].unsqueeze(-1),  # (split_size, 1)
                "target": y_val[start:end, 0].unsqueeze(-1),  # (split_size, 1)
                "c": c  # Keep conditioning unchanged
            })
        
        test_data.append({
            "input": x_test[:, 0].unsqueeze(-1),  # (split_size, 1)
            "target": y_test[:, 0].unsqueeze(-1),  # (split_size, 1)
            "c": c  # Keep conditioning unchanged
        })    
        return None

    for key in tqdm(x_files.keys()):  # Iterate over x_files
        if key in y_files:  # Ensure there's a matching y file
            c = parse_conditioning(key)
            split_audio(x_files[key], y_files[key], c, args.sequence_length)

    return train_data, val_data, test_data