import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
from visualization import plot_things
import torch
from torch import nn
import torch.optim as optim
from sklearn import metrics
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pytorch_toolbelt.losses import dice, DiceLoss
from pytorch_toolbelt.losses.functional import soft_dice_score

from userpaths import *


classes = torch.arange(0,8)

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
            inputs, targets = inputs.to(device), targets.to(device).float()
            
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
                
                val_acc = []
                train_loss.append(batch_loss.item())
                # Compute accuracies on validation set.
                with torch.no_grad():
                    model.eval()
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(device), targets.to(device).float()
                        output = model(inputs)
                        loss = loss_fn(output, targets)
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
    
    print("Finished training.")
    return train_loss, val_loss