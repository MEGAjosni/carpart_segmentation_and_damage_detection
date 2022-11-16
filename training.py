import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
from dataloader import CarDataset
from torch.utils.data import Dataset, DataLoader
from VAE_v2 import VAE_v2
from Unet_v1 import UNet
from diceloss import DiceLoss
import torch
from torch import nn
import torch.optim as optim
from sklearn import metrics
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Could be dice
def accuracy(target, pred):
    return metrics.accuracy_score(target.detach().cpu().numpy(), pred.detach().cpu().numpy())

"""class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)
        #hack temporary solution
        targets[targets > 0] = 1
        
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        
        return 1 - dice"""


user = 'Alek'

if user == 'Marcus':
    train_folder = r"C:\Users\Marcu\OneDrive - Danmarks Tekniske Universitet\DTU\Kandidat\1. Semester\Deep Learning\clean_data\train_data"
    test_folder = r"C:\Users\Marcu\OneDrive - Danmarks Tekniske Universitet\DTU\Kandidat\1. Semester\Deep Learning\clean_data\test_data"
    val_folder = r"C:\Users\Marcu\OneDrive - Danmarks Tekniske Universitet\DTU\Kandidat\1. Semester\Deep Learning\clean_data\validation_data"
elif user == 'Alek':
    train_folder = r"C:\Users\aleks\OneDrive\Skole\DTU\7. Semester\Deep Learning\clean_data\train_data"
    test_folder = r"C:\Users\aleks\OneDrive\Skole\DTU\7. Semester\Deep Learning\clean_data\test_data"
    val_folder = r"C:\Users\aleks\OneDrive\Skole\DTU\7. Semester\Deep Learning\clean_data\validation_data"
    save_folder = r"C:\Users\aleks\OneDrive\Skole\DTU\7. Semester\Deep Learning"
elif user == 'Jonas':
    folder = 'hej'

#%% Training

def train_NN(model, train_loader, val_loader, save_file='untitled', batch_size=64, num_epochs=20, validation_every_steps=500, learning_rate=0.001, loss_fn=nn.BCEWithLogitsLoss()):

    loss_fn.double()
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"  
        
    print(device)
        
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)    

    step = 0
    model.train()
    
    for epoch in tqdm(range(num_epochs)):
        
        print("epoch :", epoch)
        val_loss = []
        train_loss = []
        
        for inputs, targets in tqdm(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass, compute gradients, perform one training step.
            output = model(inputs)
            batch_loss = loss_fn(output, targets)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
    
            # Increment step counter
            step += 1
            print("step: ",step,"loss: ",batch_loss.item())
            
            
            if step % validation_every_steps == 0:
                
                train_loss.append(batch_loss.item())
                # Compute accuracies on validation set.
                with torch.no_grad():
                    model.eval()
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        output = model(inputs)
                        loss = loss_fn(output, targets)
                        
                        val_loss.append(loss.item())
                        
                    output_plot = output.cpu()
                    target_plot = targets.cpu()
                    output_plot = output_plot[0]
                    target_plot = target_plot[0]
        
                    fig, axs = plt.subplots(1,2, sharey='row',
                                        gridspec_kw={'hspace': 0, 'wspace': 0})

                    axs[0].imshow(output_plot[0].numpy(),cmap="gray")
                    axs[0].set_title('Output')
                    axs[1].imshow(target_plot[0],cmap = "gray")
                    axs[1].set_title('Segmentation mask')
                    plt.show()
        
                    model.train()
                    
                # Append average validation accuracy to list.
         
                print(f"Step {step:<5}   training loss: {batch_loss.item()}")
                print(f"             val loss: {loss.item()}")
    
    # Save model
    #path_models = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    path_models = r"C:\Users\aleks\OneDrive\Skole\DTU\7. Semester\Deep Learning"
    
    suffix = ''
    index = 0
    while save_file + suffix + '.pt' in os.listdir(path_models):
        suffix = '_' + str(index)
        index += 1
        
    torch.save(model, os.path.join(path_models, save_file + suffix + '.pt'))
    
    return train_loss, val_loss
        
    
    print("Finished training.")
 
    
#%% define VAE
import torchvision.transforms as transforms

batchsize = 16
imagewidth = 128
augmentations_train = transforms.Compose([transforms.Resize(size = imagewidth),
                                    transforms.RandomRotation((0,180)),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    ])

augmentations_val = transforms.Compose([transforms.Resize(size = imagewidth),
                                    ])
vae = VAE_v2(out_channels=1)
vae.double()
train_set = CarDataset(directory=train_folder, transform = augmentations_train, changelabel=False)
val_set = CarDataset(directory=val_folder,transform = augmentations_val,changelabel=False)

train_loader = DataLoader(dataset=train_set, batch_size=batchsize, shuffle=True)
val_loader = DataLoader(dataset=val_set, batch_size=batchsize, shuffle=True)

#%% define UNet
batchsize = 8
unet = UNet()
unet.double()

train_set = CarDataset(directory=train_folder,transform = augmentations_train,changelabel=True)
val_set = CarDataset(directory=val_folder,transform = augmentations_train,changelabel=True)

train_loader = DataLoader(dataset=train_set, batch_size=batchsize, shuffle=True)
val_loader = DataLoader(dataset=val_set, batch_size=batchsize, shuffle=True)

#%%
train_loss, val_loss = train_NN(model=vae,train_loader=train_loader,val_loader=val_loader,save_file='vae_v2',batch_size=batchsize,validation_every_steps=50, learning_rate=0.0001,num_epochs=20, loss_fn = DiceLoss())

#%% plot loss
