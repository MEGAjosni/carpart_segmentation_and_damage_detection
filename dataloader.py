import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os


class CarDataset(Dataset):
    
    def __init__(self, directory):
        # Load data
        
        x = []
        y = []
        
        for filename in os.listdir(directory):
            if filename[-4:] == ".npy":
                data = np.load(os.path.join(directory, filename))
                x.append(data[:3])
                y.append(data[3])

        self.n_samples = len(x)
        x = np.array(x)
        y = np.array(y)
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
                
    
    def __getitem__(self, index):
        # dataset[]
        return self.x[index], self.y[index]
    
    
    def __len__(self):
        # len(dataset)
        return self.n_samples


dataset = CarDataset()

batchsize = 10
dataloader = DataLoader(dataset=dataset, batch_size=batchsize, shuffle=True)
dataiter = iter(dataloader)
