# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch import nn
from collections import OrderedDict

device = "cpu"

if torch.cuda.is_available():
    device = "cuda:0"
    
print("Device:",device)

""" This model is taken from: https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/unet.py

Remember to reference!!!"""

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=8, init_features=32, padding_size = 2, kernelwidth = 8):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features,kernelwidth = kernelwidth+1, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=padding_size, stride=padding_size)
        self.encoder2 = UNet._block(features, features * 2,kernelwidth = kernelwidth//2+1, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=padding_size, stride=padding_size)
        self.encoder3 = UNet._block(features * 2, features * 4, kernelwidth = kernelwidth//4+1,name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=padding_size, stride=padding_size)
        self.encoder4 = UNet._block(features * 4, features * 8, kernelwidth = kernelwidth//4+1,name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=padding_size, stride=padding_size)

        self.bottleneck = UNet._block(features * 8, features * 16, kernelwidth = kernelwidth//4+1,name="bottleneck")
        #self.bottleneck = UNet._block(features * 4, features * 8, kernelwidth = kernelwidth//4+1,name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=padding_size, stride=padding_size
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8,kernelwidth = kernelwidth//4+1, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=padding_size, stride=padding_size
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, kernelwidth = kernelwidth//4+1,name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=padding_size, stride=padding_size
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, kernelwidth = kernelwidth//2+1,name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=padding_size, stride=padding_size
        )
        self.decoder1 = UNet._block(features * 2, features, kernelwidth = kernelwidth+1,name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        #enc4 = enc3
        bottleneck = self.bottleneck(self.pool4(enc4))
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        #dec4 = bottleneck
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        
        seg = self.conv(dec1)
        
        seg = torch.sigmoid(seg)
        
        return seg

    @staticmethod
    def _block(in_channels, features, kernelwidth, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=kernelwidth,
                            padding=(kernelwidth-1)//2,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=kernelwidth,
                            padding= (kernelwidth-1)//2,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
        

#%%

unet = UNet()
unet.double()

