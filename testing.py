import torch
import os
from dataloader import CarDataset, plot_things
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pytorch_toolbelt.losses.functional import soft_dice_score
import numpy as np
import torch.nn.functional as F

from userpaths import test_folder, model_saves

#%%
# Get path of model savefile
model_savefile = 'unet_only_photo.pt'

# Load model
#model = UNet()
#model.load_state_dict(torch.load(path+model_savefile))
model = torch.load(os.path.join(model_saves, model_savefile))
#%%
#model = vae.cpu()
# Load testdata

batchsize = 3
model.to("cpu")
model.eval()
test_data = CarDataset(test_folder,changelabel=True)
test_loader = DataLoader(dataset=test_data, batch_size=batchsize)


# Compute dice coefficients
dicecoeffs = []
for inputs,targets in test_loader:
    pred = model(inputs)
    y_pred = F.one_hot(torch.argmax(pred, dim = 1),9)
    y_pred = y_pred.permute(0, 3, 1, 2)
    targets = targets.view(batchsize, 9, -1)
    y_pred = y_pred.view(batchsize, 9, -1)
    dice_score = soft_dice_score(y_pred,targets,dims = 2)
    print("total dice :\n",dice_score)
    print("mean dice :\n",dice_score.mean())
    dicecoeffs.append(dice_score)
    
    
print("overall mean ", np.mean(np.array(dicecoeffs)))
