import os

import torch

import numpy as np

class MIDIDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.path = os.path.join(path, 'transposed')

        self.files = os.listdir(self.path)
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file = self.files[idx]
        midi = np.load(os.path.join(self.path, file))
        midi = torch.tensor(midi, dtype=torch.long)

        return midi