import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
from dataloader import CarDataset,plot_things
from torch.utils.data import Dataset, DataLoader
from VAE_v2 import VAE_v2
from VAE_v1 import VAE_v1
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
from pytorch_toolbelt.losses import dice

classes = torch.arange(0,8)

user = 'Marcus'

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
            #print(max(targets.unique()))
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass, compute gradients, perform one training step.
            output = model(inputs)
            batch_loss = loss_fn(output, targets.long())
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
                        loss = loss_fn(output, targets.long())
                        
                        val_loss.append(loss.item())
                    
                    idx = 0 #choose first element in batch
                    carpart = 0 #show door
                    plot_things(inputs,targets, prediction = output ,idx = idx, carpart = all)

        
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
import torchvision.transforms as transforms

batchsize = 8
unet = UNet(out_channels=8)
unet.double()

imagewidth = 128
augmentations_train = transforms.Compose([transforms.Resize(size = imagewidth),
                                    transforms.RandomRotation((0,180)),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    ])

augmentations_val = transforms.Compose([transforms.Resize(size = imagewidth),
                                    ])

train_set = CarDataset(directory=train_folder,transform = augmentations_train,changelabel=True)
val_set = CarDataset(directory=val_folder,transform = augmentations_train,changelabel=True)

train_loader = DataLoader(dataset=train_set, batch_size=batchsize, shuffle=True)
val_loader = DataLoader(dataset=val_set, batch_size=batchsize, shuffle=True)

#%%
train_loss, val_loss = train_NN(model=unet,train_loader=train_loader,val_loader=val_loader,save_file='unet',batch_size=batchsize,validation_every_steps=100, 
                                learning_rate=0.005,num_epochs=20, loss_fn = dice.DiceLoss("multilabel",classes))

#%% plot loss
