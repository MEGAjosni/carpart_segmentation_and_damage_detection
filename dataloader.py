import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random

class CarDataset(Dataset):
    
    def __init__(self, directory, subfolders=False, relations=None):
        '''
        Description
        -----------
        Constructor for the dataclass CarDataset
        
        
        Parameters
        ----------
        directory : str
            Directory where data is stored.
            
        subfolders : bool, optional
            True if data is divided into subfolders.
            The default is False.
        
        relations : list, optional
            Relations between number of samples in each subfolder.
            The list is sorted in alphabetical order according to the subfolder names.
            If unspecified, all data is included.
            The default is None.
        
        
        Example
        -------        
        A dataset that sample 20%, 30%, 40% and 10% from the first, second,
        third and fourth subfolder respectively, can be made in the following way:

            subfolder = True

            relations = [0.20, 0.30, 0.40, 0.10]
            or
            relations = [20, 30, 40, 10]
            or
            relations = [4, 6, 8, 2]
            etc.
        
        
        All data in all subfolders can be included in the following way:
        
            subfolder = True
            relations = None
        
        
        NOTE! the initializer will always include as much data as possible while
        maintaining the specified relation.
        
        '''
        
        
        self.dir = directory
        
        if subfolders:
            # Define types of training data and get number of samples within each category
            category_paths = [f.path for f in os.scandir(self.dir) if f.is_dir()]
            n_samples = np.array([len(os.listdir(subpath)) for subpath in category_paths])
            
            # Use all data if sampling is unspecified
            if relations is None:
                relations = n_samples
            
            # Normalize
            relations = np.array(relations) / sum(relations)
            
            # Number of samples to include from each category in training set
            sample_sizes = list(map(int, 1 / max(relations / n_samples) * relations))
            
            # Sample each category
            paths = []
            for i, subpath in enumerate(category_paths):
                for filename in random.sample(os.listdir(subpath), sample_sizes[i]):
                    paths.append(os.path.join(subpath, filename))
            
            self.data_paths = paths
        
        else:
            self.data_paths = [os.path.join(self.dir, filename) for filename in os.listdir(self.dir) if filename[-4:] == '.npy']

    
    def __getitem__(self, index):
        # dataset[]
        data = np.load(self.data_paths[index])
        return torch.tensor(data[:3]), torch.tensor(np.array([data[3]]))
    
    
    def __len__(self):
        # len(dataset)
        return len(self.data_paths)
