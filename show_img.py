import numpy as np
import matplotlib.pyplot as plt
import os
from time import sleep


folder = r"C:\Users\aleks\OneDrive\Skrivebord\clean_data\train_data\\"

for filename in os.listdir(folder):
    arr = np.load(folder+filename)
    
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
    
    