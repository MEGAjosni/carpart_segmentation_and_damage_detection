import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image


user = 'Alek'

if user == 'Marcus':
    folder  = r"C:\Users\Marcu\OneDrive - Danmarks Tekniske Universitet\DTU\Kandidat\1. Semester\Deep Learning\clean_data\train_data"
elif user == 'Alek':
    folder  = r"C:\Users\aleks\OneDrive\Skole\DTU\7. Semester\Deep Learning\clean_data\train_data"
elif user == 'Jonas':
    folder = 'hej'


imagewidth = 128


class CarDataset(Dataset):
    
    def __init__(self, directory, transform = None, changelabel = None):
        # Load data
        
        
        self.dir = directory
        datalist = os.listdir(directory)
        idx = [".npy" in string for string in datalist]
        self.datalist = [i for indx,i in enumerate(datalist) if idx[indx] == True]
        self.transform = transform
        self.changelabel = changelabel        
    
    def __getitem__(self, index):
        # dataset[]
        
        filename = os.path.join(self.dir, self.datalist[index])
        data = np.load(filename, allow_pickle = True)
        
        if self.changelabel:
            label = self.transform_segmentation_mask(9,data[3:])
            label[0] = -label[0]+1 #fix background
            data = np.concatenate((data[:3],label),axis=0)
            
            
        data = torch.tensor(data)
        if self.transform:
            data = self.transform(data)
            
        image = data[:3]
        label = data[3:]
        if self.changelabel:
            label[0] = (label[0]-1)*(-1) #fix
        return image, label
    
    
    def __len__(self):
        return len(self.datalist)
    
    def transform_segmentation_mask(self,C, label):
        N,M = label[0].shape
        mask = np.zeros((C,N,M))
        for carpart in np.unique(label):
            carpart = int(carpart)
            mask[carpart][label[0] == carpart] = 1.0
        return mask

augmentations = transforms.Compose([transforms.Resize(size = imagewidth),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomRotation((0,180)),
                                    ])
dataset = CarDataset(directory = folder, transform = augmentations,changelabel = True)

batchsize = 10

dataloader = DataLoader(dataset=dataset, batch_size=batchsize,shuffle=True)
dataiter = iter(dataloader)

images,labels = next(dataiter)

#%%
images = images.permute(0,2,3,1)
labels = labels.permute(0,2,3,1)
#%%#%% Visuallization
idx = 5
carpart =3

fig, axs = plt.subplots(1,2, sharey='row',
                    gridspec_kw={'hspace': 0, 'wspace': 0})

axs[0].imshow(images[idx].numpy())
axs[0].set_title('Image')
axs[1].imshow(labels[idx][:,:,carpart],cmap = "gray")
axs[1].set_title('Segmentation mask')


#%%
"""
from tqdm import tqdm

maxs = []
for images,labels in tqdm(dataloader):
    maxs.append(torch.max(labels).detach().numpy())
    """