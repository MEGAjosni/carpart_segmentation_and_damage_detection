import torch
import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))

from Unet_v1 import UNet
from dataloader import CarDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pytorch_toolbelt.losses.functional import soft_dice_score
import numpy as np
import torch.nn.functional as F
from userpaths import test_folder, model_saves

#%%
models = [name for name in os.listdir(model_saves) if name[-2:] in ['pt', 'pth']]

print(models)

def DiceCoefficients(
        model,
        test_loader
    ):

    dice_scores = []
    overall_dice_nobg = []
    overall_dice = []
    
    for i, data in enumerate(test_loader):
        inputs, targets = data
        pred = model(inputs)
        pred = F.one_hot(torch.argmax(pred, dim = 1),9)
        pred = pred.permute(0, 3, 1, 2)
        
        temp = [soft_dice_score(pred[:,i,:,:], targets[:,i,:,:]).item() for i in range(9)]
        dice_scores.append(temp)
        
        pred1 = pred[:,1:,:,:].view(len(inputs), 8, -1)
        targetsv1 = targets[:,1:,:,:].view(len(inputs), 8, -1)
        overall_dice_nobg.append(soft_dice_score(pred1, targetsv1).item())
        
        pred2 = pred.view(len(inputs), 9, -1)
        targetsv2 = targets.view(len(inputs), 9, -1)
        overall_dice.append(soft_dice_score(pred2, targetsv2).item())
    
    return np.mean(overall_dice), np.mean(overall_dice_nobg), np.mean(np.array(dice_scores), axis=0)



'''
#%%
# Get path of model savefile
model_savefile = 'unet_crossentropy'
model2_savefile ='unet_colorjit.pt'
# Load model
model = UNet()
#model2 = UNet()
model.load_state_dict(torch.load(model_savefile,map_location=torch.device('cpu')))
#model2.load_state_dict(torch.load(model2_savefile,map_location=torch.device('cpu')))

model2 = torch.load(os.path.join(model_saves, model2_savefile),map_location=torch.device('cpu'))

model.double()
model2.double()
#%%
#model = vae.cpu()
# Load testdata


batchsize = 1
model.to("cpu")
model.eval()
test_data = CarDataset(test_folder,changelabel=True)
test_loader = DataLoader(dataset=test_data, batch_size=batchsize, shuffle = False)

#%%
# Compute dice coefficients
dicecoeffs = []
for i,data in enumerate(test_loader):
    inputs,targets = data
    pred1 = model(inputs)
    pred2 = model2(inputs)
    plot_things(inputs,targets,predictions = [pred1,pred2], modelnames = ["Weighted CE", "Dice"], invert = True, imgidx = i)
    """
    y_pred = F.one_hot(torch.argmax(pred, dim = 1),9)
    y_pred = y_pred.permute(0, 3, 1, 2)
    targets = targets.view(batchsize, 9, -1)
    y_pred = y_pred.view(batchsize, 9, -1)
    dice_score = soft_dice_score(y_pred,targets)
    print("total dice :\n",dice_score)
    print("mean dice :\n",dice_score.mean())
    dicecoeffs.append(dice_score)
    """
#print("overall mean ", torch.stack(dicecoeffs,0).mean())
#%% plotting
inputs,targets = next(iter(test_loader))
idcs = [0,1,2]

inputs = inputs[idcs,:,:]
targets = targets[idcs,:,:]
pred1 = model(inputs)
pred2 = model2(inputs)

#%%
y_pred1 = F.one_hot(torch.argmax(pred1, dim = 1),9)
y_pred1 = y_pred1.permute(0, 3, 1, 2)
y_pred1 = y_pred1.view(len(idcs), 9, -1)

y_pred2 = F.one_hot(torch.argmax(pred2, dim = 1),9)
y_pred2 = y_pred2.permute(0, 3, 1, 2)
y_pred2 = y_pred2.view(len(idcs), 9, -1)


targetsv = targets.view(len(idcs), 9, -1)
#%%
dice_score1 = soft_dice_score(y_pred1,targetsv, dims = (1,2))
dice_score2 = soft_dice_score(y_pred2,targetsv, dims = (1,2))

dicescores = torch.stack([dice_score1,dice_score2]).T
#%%
idc = np.arange(len(idcs))
plot_things(inputs,targets,predictions = [pred1,pred2], modelnames = ["Weighted CE", "Dice"], invert = True, idx = idc,dicescores = dicescores) 

'''