import numpy as np
import scipy.signal
import torch
import os
from glob import glob

class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", lowcut=0.03, highcut=100, fs=1000) -> None:
        super().__init__()
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        
        self.split = split
        self.data_dir = data_dir
        self.num_classes = 1854
        self.num_samples = len(glob(os.path.join(data_dir, f"{split}_X", "*.npy")))
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.b, self.a = self._butter_bandpass()

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, i):
        X_path = os.path.join(self.data_dir, f"{self.split}_X", str(i).zfill(5) + ".npy")
        X = self._load_and_filter(X_path)
        X = X.copy()  
        X = torch.from_numpy(X)

        subject_idx_path = os.path.join(self.data_dir, f"{self.split}_subject_idxs", str(i).zfill(5) + ".npy")
        subject_idx = torch.from_numpy(np.load(subject_idx_path))

        if self.split in ["train", "val"]:
            y_path = os.path.join(self.data_dir, f"{self.split}_y", str(i).zfill(5) + ".npy")
            y = torch.from_numpy(np.load(y_path))

            return X, y, subject_idx
        else:
            return X, subject_idx

    @property
    def num_channels(self) -> int:
        return np.load(os.path.join(self.data_dir, f"{self.split}_X", "00000.npy")).shape[0]

    @property
    def seq_len(self) -> int:
        return np.load(os.path.join(self.data_dir, f"{self.split}_X", "00000.npy")).shape[1]

    def _butter_bandpass(self):
        nyquist = 0.5 * self.fs
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        b, a = scipy.signal.butter(5, [low, high], btype='band')
        return b, a

    def _bandpass_filter(self, data):
        return scipy.signal.filtfilt(self.b, self.a, data, axis=-1)

    def _load_and_filter(self, filepath):
        data = np.load(filepath)
        filtered_data = self._bandpass_filter(data)
        return filtered_data
