import numpy as np
import torch
from torch import nn

device = "cpu"

if torch.cuda.is_available():
    device = "cuda:0"
    
print("Device:",device)

#%% get test
image_path = r"C:\Users\aleks\OneDrive\Skole\DTU\7. Semester\Deep Learning\clean_data\train_data\0.npy"
arr = np.load(image_path)
test = torch.tensor(arr[0:3])
test = test[None,:,:,:]

print(test.shape)

#%% An actual variational auto encoder

class VAE_v2(nn.Module):
    def __init__(self, in_channels=3, out_channels=1,featureDim = 24*147*147, zDim = 256):
        super(VAE_v2,self).__init__()
    
    
        self.relu = nn.ReLU()
        #conv1
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=64)
        
        #Max Pool 1
        #self.maxpool1= nn.MaxPool2d(kernel_size=2,return_indices=True)
        
        #conv2
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=32)
        
        #Max Pool 2
        #self.maxpool2= nn.MaxPool2d(kernel_size=2,return_indices=True)
        
        #conv3
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=16)
        
        #three linear layers
        self.FC1 = nn.Linear(featureDim,zDim)
        self.FC2 = nn.Linear(featureDim,zDim)
        
        self.decFC1 = nn.Linear(zDim, featureDim)
        
        #De Convolution 1
        self.deconv1=nn.ConvTranspose2d(in_channels=24,out_channels=12,kernel_size=16)
        
        #Max UnPool 1
        #self.maxunpool1=nn.MaxUnpool2d(kernel_size=2)
        
        #De Convolution 2
        self.deconv2=nn.ConvTranspose2d(in_channels=12,out_channels=6,kernel_size=32)
        
        #Max UnPool 2
        #self.maxunpool2=nn.MaxUnpool2d(kernel_size=2)
        
        #De Convolution 3
        self.deconv3=nn.ConvTranspose2d(in_channels=6,out_channels=1,kernel_size=64)
        
    def encoder(self,x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        a,b,c,d = x.shape
        
        x = x.view(-1, b*c*d) #24*147*147
        mu = self.FC1(x)
        logVar = self.FC2(x)
        
        return mu, logVar
        
    def reparameterize(self, mu, logVar):
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std*eps
    
    def decoder(self,z):
        x = self.relu(self.decFC1(z))
        x = x.view(-1, 24, 147, 147)
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.relu(self.deconv3(x))

        return x
        
    def forward(self,x):
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        out = self.decoder(z)
        return out, mu, logVar
    
vae = VAE_v2()
vae.double()