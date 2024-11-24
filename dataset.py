# dataset.py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional

class TinyStoriesDataset(Dataset):
    def __init__(self, data_path: str, block_size: int = 256):
        """
        Args:
            data_path: Path to the binary data file (train.bin or validation.bin)
            block_size: Size of text blocks to use
        """
        self.block_size = block_size
        
        # Load the binary data
        with open(data_path, 'rb') as f:
            self.data = np.fromfile(f, dtype=np.uint16)
            
    def __len__(self):
        return len(self.data) - self.block_size
        
    def __getitem__(self, idx):
        # Get block of text and next tokens for targets
        chunk = self.data[idx:idx + self.block_size + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y

def create_dataloader(data_path: str,
                     block_size: int = 256, 
                     batch_size: int = 32,
                     num_workers: int = 4,
                     shuffle: bool = False):
    """
    Create a dataloader for TinyStories binary data
    
    Args:
        data_path: Path to binary file
        block_size: Size of text blocks
        batch_size: Batch size for dataloader
        num_workers: Number of worker processes
        shuffle: Whether to shuffle the data
    """
    dataset = TinyStoriesDataset(data_path, block_size=block_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return dataloader

if __name__ == "__main__":
    # Test the dataset and dataloader
    dataloader = create_dataloader(
        data_path='validation.bin',
        block_size=256,
        batch_size=32
    )
    
    # Print a sample
    for batch_x, batch_y in dataloader:
        print(f"Input shape: {batch_x.shape}")
        print(f"Target shape: {batch_y.shape}")
        break