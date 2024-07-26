import array
import torch


def dataFileInput(filename):
    
    with open(filename, "rb") as file:
        IQdata = array.array("f",file.read())
    numSamples = int(len(IQdata)/2)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Data on GPU")
    else:
        device = torch.device('cpu')
        print("Data on CPU")
    
    #detach the tensor
    data = torch.empty([1,numSamples], dtype=torch.complex64, device=device, requires_grad=False)
    data = data.detach()

    for k in range(0,numSamples,1):
        data[0,k] = IQdata[2*k] + 1j*IQdata[2*k+1]
    
    return data







