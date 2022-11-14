import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import torchvision.transforms as transforms
from PIL import Image


user = 'Marcus'

if user == 'Marcus':
    folder  = r"C:\Users\Marcu\OneDrive - Danmarks Tekniske Universitet\DTU\Kandidat\1. Semester\Deep Learning\clean_data\train_data"
elif user == 'Alek':
    folder  = r"C:\Users\aleks\OneDrive\Skole\DTU\7. Semester\Deep Learning\clean_data\train_data"
elif user == 'Jonas':
    folder = 'hej'


imagewidth = 128


class CarDataset(Dataset):
    
    def __init__(self, directory, transform = None):
        # Load data
        
        
        self.dir = directory
        datalist = os.listdir(directory)
        idx = [".npy" in string for string in datalist]
        self.datalist = [i for indx,i in enumerate(datalist) if idx[indx] == True]
        self.transform = transform
        
    
    def __getitem__(self, index):
        # dataset[]
        
        filename = os.path.join(self.dir, self.datalist[index])
        data = np.load(filename, allow_pickle = True)
        
        data = torch.tensor(data)
        if self.transform:
            data = self.transform(data)
            
        image = data[:3]
        label = data[3:]
        return image, label
    
    
    def __len__(self):
        # len(dataset)
        return len(self.datalist)

augmentations = transforms.Compose([transforms.RandomRotation((0,180)),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.Resize(size = imagewidth),
                                    ])
dataset = CarDataset(directory = folder, transform = augmentations)

batchsize = 10

dataloader = DataLoader(dataset=dataset, batch_size=batchsize,shuffle=True)
dataiter = iter(dataloader)


#%% Visuallization
import matplotlib.pyplot as plt

images,labels = next(dataiter)

images = images.permute(0,2,3,1)
#%%
idx = 6

fig, axs = plt.subplots(1,2, sharey='row',
                    gridspec_kw={'hspace': 0, 'wspace': 0})

axs[0].imshow(images[idx].numpy())
axs[0].set_title('Image')
axs[1].imshow(labels[idx][0],cmap = "gray")
axs[1].set_title('Segmentation mask')