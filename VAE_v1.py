import numpy as np
import torch
from torch import nn

device = "cpu"

if torch.cuda.is_available():
    device = "cuda:0"
    
print("Device:",device)

#%% Alexander shitty net

class VAE(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(VAE,self).__init__()
    
    
        self.relu = nn.ReLU()
        #conv1
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=4)
        
        #Max Pool 1
        self.maxpool1= nn.MaxPool2d(kernel_size=2,return_indices=True)
        
        #conv2
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=4)
        
        #Max Pool 2
        self.maxpool2= nn.MaxPool2d(kernel_size=2,return_indices=True)
        
        #conv3
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=4)
        
        #De Convolution 1
        self.deconv1=nn.ConvTranspose2d(in_channels=24,out_channels=12,kernel_size=4)
        
        #Max UnPool 1
        self.maxunpool1=nn.MaxUnpool2d(kernel_size=2)
        
        #De Convolution 2
        self.deconv2=nn.ConvTranspose2d(in_channels=12,out_channels=6,kernel_size=4)
        
        #Max UnPool 2
        self.maxunpool2=nn.MaxUnpool2d(kernel_size=2)
        
        #De Convolution 3
        self.deconv3=nn.ConvTranspose2d(in_channels=6,out_channels=1,kernel_size=4)
        
    def forward(self,x):
        out=(self.conv1(x))
        size1 = out.size()
        out,indices1=self.maxpool1(out)
        out=self.relu(self.conv2(out))
        size2 = out.size()
        out,indices2=self.maxpool2(out)
        out=self.relu(self.conv3(out))
 
        out=self.relu(self.deconv1(out))
        out=self.maxunpool1(out,indices2,size2)
        out=self.relu(self.deconv2(out))
        out=self.maxunpool2(out,indices1,size1)
        out=self.relu(self.deconv3(out))
        
        
        return(out)
    
vae = VAE(in_channels=3,out_channels=1)
vae.double()

#%% Marcus CHAD net

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, 
                 num_classes, #number of output classes
                 kernw = 5, #width in first convolutional layer
                 kernchannels = 6, #number of kernels in first conv layer
                 imagew = 32, #width of image
                 channels = 3, #channels in input image,
                 poolingsize = 2, #size of maxpool
                 ):
        super(Net, self).__init__()
        self.num_classes = num_classes


        # Your code here!

        self.conv1 = nn.Conv2d(in_channels = channels,
                               out_channels = kernchannels,
                               kernel_size = kernw)
        n1 = imagew-kernw #size after conv1

        self.pool = nn.MaxPool2d(kernel_size = poolingsize)
        n1 //=2; n1 +=1 # image size after pooling 

        outchannels = kernchannels*3 #increase amount of channels in second conv layer
        self.conv2 = nn.Conv2d(in_channels = kernchannels,
                               out_channels = outchannels,
                               kernel_size = kernw)
        
        n1 -= kernw #size after conv2
        n1+= 1
        #n1 //= 2;  #size after second pooling

        l1in = n1*n1*outchannels #size after vectorization
        self.l1 = nn.Linear(l1in, l1in//2)
        self.l2 = nn.Linear(l1in//2, l1in//4)
        self.l3 = nn.Linear(l1in//4, num_classes)
        self.drop = nn.Dropout(0.4)
        
    def forward(self, x):
        #conv layers
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        #x = self.pool(x)
        #dense layers
        x = torch.flatten(x,1)
        x = F.relu(self.l1(x))
        x = self.drop(x)
        x = F.relu(self.l2(x))
        x = self.drop(x)
        x = self.l3(x)
        return x

#%%

net = Net(num_classes = 10, #number of output classes
          kernw = 50, #width in first convolutional layer
          kernchannels = 6, #number of kernels in first conv layer
          imagew = 256, #width of image
          channels = 3, #channels in input image,
          poolingsize = 2, #size of maxpool
          )
net.double()

#%%

net(images)
        
        