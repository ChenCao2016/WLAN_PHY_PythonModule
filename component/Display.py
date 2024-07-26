from matplotlib import pyplot as plt
import numpy as np
import torch


def absLinear(data:torch.tensor):
    fig, ax1 = plt.subplots(nrows=1, ncols=1)
    ax1.plot(np.abs(data.to('cpu').numpy()[0,:]))
    ax1.grid()
    fig.show()

def absLog(data:torch.tensor):
    fig, ax1 = plt.subplots(nrows=1, ncols=1)
    ax1.plot(20*np.log10(np.abs(data.to('cpu').numpy()[0,:])))
    ax1.grid()
    fig.show()