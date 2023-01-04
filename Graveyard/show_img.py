import numpy as np
import matplotlib.pyplot as plt
import os
from time import sleep
import torch

from userpaths import train_folder

for filename in os.listdir(os.path.join(train_folder, 'photo')):
    arr = np.load(os.path.join(train_folder, 'photo', filename))
    
    fig, axs = plt.subplots(2, 2,sharex='col', sharey='row',
                        gridspec_kw={'hspace': 0, 'wspace': 0})
    axs[0,0].imshow(arr[0])
    axs[0,0].set_title('First Channel')
    axs[0,1].imshow(arr[1])
    axs[0,1].set_title('Second Channel')
    axs[1,0].imshow(arr[2])
    axs[1,0].set_title('Third Channel')
    axs[1,1].imshow(arr[3])
    axs[1,1].set_title('Fourth Channel')
    
    plt.show()
    
    break
    #sleep(2)
    
    
#%%
test = torch.tensor(arr[0:3])
test = test[None,:,:,:]
print(test.shape)