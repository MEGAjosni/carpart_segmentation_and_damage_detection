import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn.functional as F
import random

# Fix error at marcus pc
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#imagewidth = 256
class CarDataset(Dataset):
    
    def __init__(self, directory, transform = None, changelabel = None, subfolders=False, relations=None):
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

        '''
        self.dir = directory
        datalist = os.listdir(directory)
        idx = [".npy" in string for string in datalist]
        self.datalist = [i for indx,i in enumerate(datalist) if idx[indx] == True]
        '''
        
        self.transform = transform
        self.changelabel = changelabel
    
    def __getitem__(self, index):
        # dataset[]
        
        data = np.load(self.data_paths[index], allow_pickle=True)
        
        label = self.transform_segmentation_mask(9,data[3:])
        label[0] = (label[0]-1)*(-1) if self.changelabel else label[0]#invert background
        data = np.concatenate((data[:3],label),axis=0)
            
        data = torch.tensor(data)
        if self.transform:
            data = self.transform(data)
            
        image = data[:3]
        label = data[3:]
        
        if not self.changelabel:
            mask = torch.zeros(1,label.shape[1],label.shape[2])
            for i,segmentation in enumerate(label):
                mask[0] += segmentation*(i+1)
            label= mask
        
        label[0] = (label[0]-1)*(-1) if self.changelabel else label[0]
        return image, label.round().int()
    
    
    def __len__(self):
        return len(self.data_paths)
    
    def transform_segmentation_mask(self,C, label):
        N,M = label[0].shape
        mask = np.zeros((C,N,M))
        for carpart in np.unique(label):
            carpart = int(carpart)
            mask[carpart][label[0] == carpart] = 1.0
        return mask
    '''
    augmentations = transforms.Compose([#transforms.Resize(size = imagewidth),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomRotation((-30,30)),
                                        ])
    dataset = CarDataset(directory = folder, transform = augmentations,changelabel = True)
    
    batchsize = 3
    
    dataloader = DataLoader(dataset=dataset, batch_size=batchsize,shuffle=True)
    dataiter = iter(dataloader)
    
    images,labels = next(dataiter)
    '''

def plot_things(images,labels,predictions = [], idx = 0, carpart = all):
    image = images[idx].permute(1,2,0)
    label = labels[idx]
    label = torch.argmax(label,dim = 0) if carpart == all else label[carpart,:,:]
    
    n = 3 if len(predictions) > 0 else 2
    fig, axs = plt.subplots(1,n, sharey='row',
                        gridspec_kw={'hspace': 0, 'wspace': 0})
    
    axs[0].imshow(image.numpy())
    axs[0].set_title('Image')
    axs[1].imshow(label,cmap = "gray")
    axs[1].set_title('Ground truth')
    axs[0].set_axis_off()
    axs[1].set_axis_off()
    if n > 2:
        prediction = predictions[idx]
        prediction = torch.argmax(prediction,dim = 0) if carpart == all else prediction[carpart,:,:]
        axs[2].imshow(prediction.detach().numpy(),cmap = "gray")
        axs[2].set_title('Prediction')
        axs[2].set_axis_off()
    plt.show()
