import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

user = 'Marcus'

if user == 'Marcus':
    folder  = r"C:\Users\Marcu\OneDrive - Danmarks Tekniske Universitet\DTU\Kandidat\1. Semester\Deep Learning\clean_data\train_data"
elif user == 'Alek':
    folder  = r"C:\Users\aleks\OneDrive\Skrivebord\clean_data\train_data"
elif user == 'Jonas':
    folder = 'hej'


class CarDataset(Dataset):
    
    def __init__(self, directory):
        # Load data
        
        
        self.dir = directory
        datalist = os.listdir(directory)
        idx = [".npy" in string for string in datalist]
        self.datalist = [i for indx,i in enumerate(datalist) if idx[indx] == True]
        
        """
        x = []
        y = []
        
        for filename in os.listdir(directory):
            if filename[-4:] == ".npy":
                data = np.load(os.path.join(self, filename))
                x.append(data[:3])
                y.append(data[3])
                

        self.n_samples = len(x)
        x = np.array(x)
        y = np.array(y)
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
                """
    
    def __getitem__(self, index):
        # dataset[]
        
        filename = os.path.join(self.dir, self.datalist[index])
        data = np.load(filename, allow_pickle = True)

        return torch.tensor(data[:3]), torch.tensor(np.array([data[3]]))
    
    
    def __len__(self):
        # len(dataset)
        return len(self.datalist)


dataset = CarDataset(directory = folder)

batchsize = 10
dataloader = DataLoader(dataset=dataset, batch_size=batchsize, shuffle=True)
dataiter = iter(dataloader)

#%%

images,labels = next(dataiter)