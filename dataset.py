# dataset.py
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, input_tensor, target_tensor):
        self.input_tensor = input_tensor
        self.target_tensor = target_tensor

    def __len__(self):
        return len(self.input_tensor)

    def __getitem__(self, index):
        x = self.input_tensor[index]
        y = self.target_tensor[index]
        return x, y
    
    def add_instance(self, new_input, new_target):
        new_input = torch.Tensor(new_input)
        new_target = torch.Tensor(new_target)
        self.input_tensor = torch.cat([self.input_tensor, new_input])
        self.target_tensor = torch.cat([self.target_tensor, new_target])

