import torch
import os
from dataloader import CarDataset, plot_things
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pytorch_toolbelt.losses.functional import soft_dice_score
import numpy as np


#%%
# Get path of model savefile
model_savefile = 'unet.pt'
path = r"C:\Users\aleks\OneDrive\Skole\DTU\7. Semester\Deep Learning\\"

# Load model
model = torch.load(path+model_savefile)
model.cpu()
#%%
#model = vae.cpu()
# Load testdata
#data_path = r"C:\Users\aleks\OneDrive\Skole\DTU\7. Semester\Deep Learning\clean_data\test_data"


model = unet
model.to("cpu")
model.eval()
test_data = CarDataset(test_folder,changelabel=True)
test_loader = DataLoader(dataset=test_data, batch_size=1)



# Compute dice coefficients
dicecoeffs = []
for inputs,targets in test_loader:
    pred = model(inputs)
    y_pred = F.one_hot(torch.argmax(pred, dim = 1),9)
    y_pred = y_pred.permute(0, 3, 1, 2)
    dice_score = []
    for i,carpart in enumerate(targets[0]): #only if batchsize = 1. # loops over carparts in targets[0]
        dice_score_eachpart = soft_dice_score(y_pred[0][i],carpart)
        print("part ",i," with dice ",dice_score_eachpart.item())
        dice_score.append(dice_score_eachpart)
    dice_score = torch.stack(dice_score,0).mean()
    print("mean dice ",dice_score.item())
    dicecoeffs.append(dice_score)
    
    
print("overall mean ", np.mean(dicecoeffs))
