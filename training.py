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
from pytorch_toolbelt.losses.functional import soft_dice_score
from dicelossJAM import DiceLossJAM
from userpaths import *


classes = torch.arange(0,8)
weights = torch.tensor([0.2,1,1,1,1,1,1,1,1])

#%% Training

def train_NN(model, train_loader, val_loader, save_folder='', save_file='untitled', batch_size=64, num_epochs=20, validation_every_steps=500, learning_rate=0.001, loss_fn=nn.BCEWithLogitsLoss()):

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
            batch_loss = loss_fn(output, targets.float())
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            # Increment step counter
            step += 1
            print("step: ",step,"loss: ",batch_loss.item())
            
            
            if step % validation_every_steps == 0:
                
                val_acc = []
                train_loss.append(batch_loss.item())
                # Compute accuracies on validation set.
                with torch.no_grad():
                    model.eval()
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        output = model(inputs)
                        loss = loss_fn(output, targets.float())
                        #accuracy
                        y_pred = F.one_hot(torch.argmax(output, dim = 1),9)
                        y_pred = y_pred.permute(0, 3, 1, 2)
                        dice_score = soft_dice_score(y_pred,targets)
                        val_acc.append(dice_score)
                        
                        val_loss.append(loss.item())
                    
                    idx = 0 #choose first element in batch
                    print("dice score: ", val_acc[-1])
                    plot_things(inputs,targets, output ,idx = idx, carpart = all)
                    model.train()
                    
                # Append average validation accuracy to list.
         
                print(f"Step {step:<5}   training loss: {batch_loss.item()}")
                print(f"             val loss: {loss.item()}")
    
    suffix = ''
    index = 0
    while save_file + suffix + '.pt' in os.listdir(model_saves):
        suffix = '_' + str(index)
        index += 1
    
    torch.save(model, os.path.join(model_saves, save_file + suffix + '.pt'))
    
    return train_loss, val_loss
        
    
    print("Finished training.")
 
    



#%% define UNet
import torchvision.transforms as transforms

batchsize = 8
unet = UNet(out_channels=9)
unet.double()

imagewidth = 256
augmentations_train = transforms.Compose([#transforms.Resize(size = imagewidth),
                                    transforms.RandomRotation((-30,30)),
                                    #transforms.RandomHorizontalFlip(p=0.5),
                                    ])

augmentations_val = transforms.Compose([transforms.Resize(size = imagewidth),
                                    ])

train_set = CarDataset(directory=train_folder, subfolders=True, relations=None,transform = None,changelabel=True,colorjit=True)
val_set = CarDataset(directory=validation_folder,transform = None,changelabel=True)

train_loader = DataLoader(dataset=train_set, batch_size=batchsize, shuffle=True)
val_loader = DataLoader(dataset=val_set, batch_size=batchsize, shuffle=True)


#%%
train_loss, val_loss = train_NN(model=unet,train_loader=train_loader,val_loader=val_loader,save_file='unet_colorjit2',batch_size=batchsize,validation_every_steps=300,
                                learning_rate=0.005,num_epochs=20, loss_fn = dice.DiceLoss("multilabel",classes, from_logits=False,smooth=1.0))

#%% plot loss
