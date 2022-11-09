import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
from dataloader import CarDataset
from torch.utils.data import Dataset, DataLoader
from VAE_v1 import VAE
import torch
from torch import nn
import torch.optim as optim
from sklearn import metrics
import numpy as np

# Could be dice
def accuracy(target, pred):
    return metrics.accuracy_score(target.detach().cpu().numpy(), pred.detach().cpu().numpy())

train_folder = r"C:\Users\aleks\OneDrive\Skrivebord\clean_data\train_data"
test_folder = r"C:\Users\aleks\OneDrive\Skrivebord\clean_data\test_data"

#%% Training

def train_NN(model, train_loader, test_loader, batch_size=64, num_epochs=100, validation_every_steps=500, learning_rate=0.001, loss_fn=nn.BCEWithLogitsLoss()):

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"  
        
    print(device)
        
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)    

    step = 0
    model.train()
    
    for epoch in range(num_epochs):
        
        test_loss = []
        train_loss = []
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass, compute gradients, perform one training step.
            output = model(inputs)
            batch_loss = loss_fn(output, targets)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
    
            # Increment step counter
            step += 1
            print(step,batch_loss)
            
            train_loss.append(batch_loss)
            
            if step % validation_every_steps == 0:
                
                
                # Compute accuracies on validation set.
                with torch.no_grad():
                    model.eval()
                    for inputs, targets in test_loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        output = model(inputs)
                        loss = loss_fn(output, targets)
                        
                        test_loss.append(loss)
        
    
                    model.train()
                    
                # Append average validation accuracy to list.
         
                print(f"Step {step:<5}   training loss: {batch_loss[-1]}")
                print(f"             test loss: {loss[-1]}")
    
    print("Finished training.")
        
    
    
    
#%%
batchsize = 16

vae = VAE()
vae.double()
train_set = CarDataset(directory=train_folder)
test_set = CarDataset(directory=test_folder)

train_loader = DataLoader(dataset=train_set, batch_size=batchsize, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batchsize, shuffle=True)

#%%
train_NN(vae,train_loader,test_loader,batch_size=batchsize,validation_every_steps=25)