import numpy as np
import torch
from torch import nn

device = "cpu"

if torch.cuda.is_available():
    device = "cuda:0"
    
print("Device:",device)

#%% Alexander shitty net

class VAE_v1(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(VAE_v1,self).__init__()
    
    
        self.relu = nn.ReLU()
        #conv1
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=16)
        
        #Max Pool 1
        self.maxpool1= nn.MaxPool2d(kernel_size=2,return_indices=True)
        
        #conv2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=24, kernel_size=8)
        
        #Max Pool 2
        self.maxpool2= nn.MaxPool2d(kernel_size=2,return_indices=True)
        
        #conv3
        self.conv3 = nn.Conv2d(in_channels=24, out_channels=32, kernel_size=4)
        
        #De Convolution 1
        self.deconv1=nn.ConvTranspose2d(in_channels=32,out_channels=24,kernel_size=4)
        
        #Max UnPool 1
        self.maxunpool1=nn.MaxUnpool2d(kernel_size=2)
        
        #De Convolution 2
        self.deconv2=nn.ConvTranspose2d(in_channels=24,out_channels=16,kernel_size=8)
        
        #Max UnPool 2
        self.maxunpool2=nn.MaxUnpool2d(kernel_size=2)
        
        #De Convolution 3
        self.deconv3=nn.ConvTranspose2d(in_channels=16,out_channels=out_channels,kernel_size=16)
        
    def forward(self,x):
        out=self.relu(self.conv1(x))
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
        out=self.deconv3(out)
        #out = self.relu(out)
        
        return(out)
    



        