'''
Functions for visualization
'''
import matplotlib.pyplot as plt
import numpy as np

def ShowTest(data, model):
    image, target = data
    pred = model(image[None, :])
    
    fig, ax = plt.subplots(1, 3)
    target = target * np.arange(9)[:,None,None]
    
    # Image
    ax[0].imshow(image.permute(1,2,0))
    ax[0].set_axis_off()
    ax[0].set_title('Image')
    
    # Target
    ax[1].imshow(target.sum(0))
    ax[1].set_axis_off()
    ax[1].set_title('Target')
    
    # Prediction
    ax[2].imshow(pred[0,:,:,:].sum(0).detach().numpy())
    ax[2].set_axis_off()
    ax[2].set_title('Prediction')
    
    plt.show()
    
    