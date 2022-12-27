'''
Functions for visualization
'''
import torch
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


def plot_things(images,labels,predictions = [], modelnames = [],idx = [0], carpart = all, invert = False, dicescores = [], imgidx = ''):

    m = len(predictions)+2 if len(predictions) > 0 else 2
    n = len(idx)
    fig, axs = plt.subplots(n,m)
    axs = axs.reshape(n,m)
    if m > 2:
        for i in range(n):
            image = images[idx[i]].permute(1,2,0)
            label = labels[idx[i]]
            label = torch.argmax(label,dim = 0) if carpart == all else label[carpart,:,:]
            label = (label-8)*(-1) if invert else label
            
            axs[i,0].imshow(image.cpu().numpy())
            axs[i,1].imshow(label.cpu(),cmap = "gray")

            axs[i,0].set_axis_off()
            axs[i,1].set_axis_off()
            for j in range(m-2):
                prediction = predictions[j][idx[i]] if type(predictions) == list else predictions[idx[i]]
                prediction = torch.argmax(prediction,dim = 0) if carpart == all else prediction[carpart,:,:]
                prediction = (prediction-8)*(-1) if invert else prediction
                axs[i,j+2].imshow(prediction.cpu(),cmap = "gray")
                #axs[i,j+2].legend("Dice: "+str(round(dicescores[i,j].item(),2)),loc = 'lower center')
                if len(dicescores) > 0:
                    axs[i,j+2].text(20, 275, "Dice: "+str(round(dicescores[i,j].item(),2)), bbox={'facecolor': 'white', 'pad': 2})


                modelname = modelnames[j] if len(modelnames) > 0 else 'Prediction '+str(j+1)
                axs[0,j+2].set_title(modelname)
                axs[i,j+2].set_axis_off()
                
    axs[0,0].set_title('Image'+str(imgidx))
    axs[0,1].set_title('Ground truth')
    fig.subplots_adjust(wspace = 0)
    plt.show()