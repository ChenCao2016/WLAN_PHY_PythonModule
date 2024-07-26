import torch
import time
from .Sync import powerDection as pd
import numpy as np
from matplotlib import pyplot as plt


def powerDetection(data:torch.tensor, powerThreshold:float):
    startTime = time.time()

    dataAbs = torch.abs(data)[0,:].numpy()

    result = np.zeros(len(dataAbs),dtype=np.int32)

    pd(dataAbs,result,int(len(dataAbs)),float(powerThreshold),int(50))

    endTime = time.time()
    print("powerDetection: {}".format(endTime-startTime))

    fig, ax1 = plt.subplots(nrows=1, ncols=1)
    ax1.plot(result)
    ax1.grid()
    fig.show()


    return np.where(result == 1)

def LSTFcorrelate(data:torch.tensor):
    result = torch.cov(data[0,:])
    pass
