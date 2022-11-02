import numpy as np
import matplotlib.pyplot as plt
import os
from time import sleep
from PIL import Image


folder = r"C:\Users\aleks\OneDrive\Skrivebord\clean_data\train_data\\"

def numpy2pil(np_array: np.ndarray) -> Image:
    """
    Convert an HxWx3 numpy array into an RGB Image
    """

    assert_msg = 'Input shall be a HxWx3 ndarray'
    assert isinstance(np_array, np.ndarray), assert_msg
    assert len(np_array.shape) == 3, assert_msg
    assert np_array.shape[2] == 3, assert_msg

    img = Image.fromarray(np_array, 'RGB')
    return img

for filename in os.listdir(folder):
    arr = np.load(folder+filename)
    arr = arr[0:3]
    arr = np.transpose(arr, (1,2,0))
    
    plt.imshow(arr.astype(np.uint8))
    plt.show()
    
    #img = Image.fromarray(arr, 'RGB')
    #img.show()
    
    break
    sleep(2)
    
    