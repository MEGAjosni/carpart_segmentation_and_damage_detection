import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
from dataloader import CarDataset
from torch.utils.data import Dataset, DataLoader
from VAE_v2 import VAE_v2
from Unet_v1 import UNet
import torch
from torch import nn
import torch.optim as optim
from sklearn import metrics
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

# Could be dice
def accuracy(target, pred):
    return metrics.accuracy_score(target.detach().cpu().numpy(), pred.detach().cpu().numpy())

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice


user = 'Alek'

if user == 'Marcus':
    train_folder = r"C:\Users\Marcu\OneDrive - Danmarks Tekniske Universitet\DTU\Kandidat\1. Semester\Deep Learning\clean_data\train_data"
    test_folder = r"C:\Users\Marcu\OneDrive - Danmarks Tekniske Universitet\DTU\Kandidat\1. Semester\Deep Learning\clean_data\test_data"
elif user == 'Alek':
    train_folder = r"C:\Users\aleks\OneDrive\Skole\DTU\7. Semester\Deep Learning\clean_data\train_data"
    test_folder = r"C:\Users\aleks\OneDrive\Skole\DTU\7. Semester\Deep Learning\clean_data\test_data"
    val_folder = r"C:\Users\aleks\OneDrive\Skole\DTU\7. Semester\Deep Learning\clean_data\validation_data"
elif user == 'Jonas':
    folder = 'hej'

#%% Training

def train_NN(model, train_loader, test_loader, val_loader, batch_size=64, num_epochs=20, validation_every_steps=500, learning_rate=0.001, loss_fn=nn.BCEWithLogitsLoss()):

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
            
            train_loss.append(batch_loss.item())
            
            if step % validation_every_steps == 0:
                
                
                # Compute accuracies on validation set.
                with torch.no_grad():
                    model.eval()
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        output = model(inputs)
                        loss = loss_fn(output, targets)
                        
                        val_loss.append(loss.item())
        
                    
                    #plotfun(images,labels,1)
                    #vae.cuda()
        
                    model.train()
                    
                # Append average validation accuracy to list.
         
                print(f"Step {step:<5}   training loss: {batch_loss.item()}")
                print(f"             test loss: {loss.item()}")
    
    print("Finished training.")
 
    
#%% define VAE
batchsize = 16

vae = VAE_v2()
vae.double()
train_set = CarDataset(directory=train_folder)
val_set = CarDataset(directory=val_folder)

train_loader = DataLoader(dataset=train_set, batch_size=batchsize, shuffle=True)
val_loader = DataLoader(dataset=val_set, batch_size=batchsize, shuffle=True)

#%% define UNet
batchsize = 16
unet = UNet()
unet.double()

train_set = CarDataset(directory=train_folder)
val_set = CarDataset(directory=val_folder)

train_loader = DataLoader(dataset=train_set, batch_size=batchsize, shuffle=True)
val_loader = DataLoader(dataset=val_set, batch_size=batchsize, shuffle=True)

#%%
train_NN(vae,train_loader,test_loader,val_loader,batch_size=batchsize,validation_every_steps=25,loss_fn = DiceLoss(), learning_rate=0.001)

#%%
import matplotlib.pyplot as plt


images,labels = next(iter(test_loader))
idx = 5
def plotfun(images,labels,idx):
    fig, axs = plt.subplots(1,2,sharex='col', sharey='row',
                        gridspec_kw={'hspace': 0, 'wspace': 0})
    axs[0].imshow(labels[idx][0])
    axs[0].set_title('True')
    with torch.no_grad():
        vae.cpu()
        pred = vae(images)[idx][0]
        axs[1].imshow(pred.numpy())
        axs[1].set_title('Prediction')
        plt.show()
        
plotfun(images,labels,idx)