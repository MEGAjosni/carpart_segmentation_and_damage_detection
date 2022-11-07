import numpy as np
import torch
from torch import nn

device = "cpu"

if torch.cuda.is_available():
    device = "cuda:0"
    
print("Device:",device)

class VAE(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(VAE,self).__init__()
        
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
        self.deconv1=nn.ConvTranspose2d(in_channels=12,out_channels=6,kernel_size=4)
        
    def forward(self,x):
        #out=nn.ReLU(self.conv1(x))
        out=self.conv1(x)
        size1 = out.size()
        out,indices1=self.maxpool1(out)
        out=nn.ReLU(self.conv2(out))
        size2 = out.size()
        out,indices2=self.maxpool2(out)
        out=nn.ReLU(self.conv3(out))
 
        out=nn.ReLU(self.deconv1(out))
        out=self.maxunpool1(out,indices2,size2)
        out=nn.ReLU(self.deconv2(out))
        out=self.maxunpool2(out,indices1,size1)
        out=nn.ReLU(self.deconv3(out))
        return(out)
    
vae = VAE(in_channels=3,out_channels=1)

        
        
        
        