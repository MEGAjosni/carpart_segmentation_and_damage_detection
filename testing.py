import torch
import os
from dataloader import CarDataset, plot_things
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pytorch_toolbelt.losses.functional import soft_dice_score
import numpy as np
import torch.nn.functional as F

#%%
# Get path of model savefile
model_savefile = 'unet_only_photo.pt'
path = r"C:\Users\aleks\OneDrive\Skole\DTU\7. Semester\Deep Learning\\"
path = r"C:\Users\Marcu\OneDrive - Danmarks Tekniske Universitet\DTU\Kandidat\1. Semester\Deep Learning\carpart_segmentation_and_damage_detection\\"
path = r'C:\Users\jonas\OneDrive\Desktop\DeepLearningProject\models'
# Load model
#model = UNet()
#model.load_state_dict(torch.load(path+model_savefile))
model = torch.load(os.path.join(path, model_savefile))
#%%
#model = vae.cpu()
# Load testdata
data_path = r"C:\Users\jonas\OneDrive\Desktop\DeepLearningProject\data\clean_data\test_data"


batchsize = 3
#model = unet
model.to("cpu")
model.eval()
test_data = CarDataset(data_path,changelabel=True)
test_loader = DataLoader(dataset=test_data, batch_size=batchsize)


# Compute dice coefficients
dicecoeffs = []
for inputs,targets in test_loader:
    pred = model(inputs)
    y_pred = F.one_hot(torch.argmax(pred, dim = 1),9)
    y_pred = y_pred.permute(0, 3, 1, 2)
    dice_score = torch.tensor([])
    for batch in range(batchsize):
        for i,carpart in enumerate(targets[batch]): #only if batchsize = 1. # loops over carparts in targets[0]
            dice_score_eachpart = soft_dice_score(y_pred[batch][i],carpart)
            print("part ",i," with dice ",dice_score_eachpart.item())
            dice_score = torch.cat((dice_score, torch.tensor([dice_score_eachpart])), 0)
    print("mean dice ",dice_score.mean())
    dicecoeffs.append(dice_score)
    
    
print("overall mean ", np.mean(dicecoeffs))
