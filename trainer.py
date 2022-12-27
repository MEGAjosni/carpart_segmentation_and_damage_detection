import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

from userpaths import *
import torch
import torchvision.transforms as transforms
from UNet import UNet
from torch.utils.data import DataLoader
from dataloader import CarDataset
from pytorch_toolbelt.losses.dice import DiceLoss
from train_loop import train_NN


imagewidth = 256
batchsize = 8

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"

model = UNet()
model.double()


augmentations_train = transforms.Compose(
    [
        #transforms.Resize(size = imagewidth),
        #transforms.ColorJitter(brightness=.5, hue=.3),
        transforms.RandomRotation((-30,30)),
        transforms.RandomHorizontalFlip(p=0.5),
    ]
)

augmentations_val = transforms.Compose(
    [
        transforms.Resize(size = imagewidth),
    ]
)

relations = [0,0,0,1]


train_loader = DataLoader(dataset=CarDataset(directory=train_folder, subfolders=True, relations=relations,transform = augmentations_train,changelabel=True, colorjit = True), batch_size=batchsize, shuffle=True)
val_loader = DataLoader(dataset=CarDataset(directory=validation_folder,transform = augmentations_train,changelabel=True), batch_size=batchsize, shuffle=True)

weight = torch.tensor([1,2,10,10,10,10,10,10,10]).float().to(device)

train_loss, val_loss = train_NN(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    save_file='unet_only_photo',
    batch_size=batchsize,
    validation_every_steps=200,
    learning_rate=0.005,num_epochs=20,
    loss_fn = DiceLoss("multilabel", from_logits = False, smooth = 1.0)
)
