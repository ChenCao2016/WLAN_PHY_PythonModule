from math import *

def Interleaver(input,output,N_CBPS):
    ret = 0;

    output.clear();

    s = max(int(N_CBPS/2),1);
    
    for j in range(len(input)):

        i = s * floor(j/s) + (j + floor(16*j/N_CBPS)) % s;
        k = 16 * i - (N_CBPS -1)*floor(16*i/N_CBPS);
        
        output.append(input[k]);

    return;


def Deinterleaver(input,output,N_CBPS):
    ret = 0;

    output.clear();

    s = max(int(N_CBPS/2),1);
    
    for k in range(len(input)):
        
        i = (N_CBPS/16)*(k % 16) + floor(k/16);
        j = s * floor(i/s) +(i + N_CBPS - floor(16*i/N_CBPS)) % s;

        output.append(input[j]);

    return;