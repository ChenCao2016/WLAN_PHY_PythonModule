import math
import numpy as np

def Interleaver(input, N_CBPS):

    output = []

    s = max(int(N_CBPS/2), 1)
    size = len(input)

    for j in range(size):

        i = s * math.floor(j/s) + (j + math.floor(16*j/size)) % s
        k = 16 * i - (size - 1)*math.floor(16*i/size)

        output.append(input[k])

    return output


def Deinterleaver(input, N_CBPS, output):
    ret = 0

    output.clear()

    s = int(max(int(N_CBPS/2), 1))
    size = len(input)

    for k in range(size):

        i = int((size/16)*(k % 16) + math.floor(k/16))
        j = int(s * math.floor(i/s) + (i + size - math.floor(16*i/size)) % s)

        output.append(input[j])

    return


class BCCDeinterleaver:
    N_CBPSSI = 10  # Number of coded bits per symbol per spatial stream per BCC interleaver block
    N_BPSCS = 1 # Number of coded bits per subcarrier per spatial stream
    iss = 1 # is the spatial stream index on which this interleaver is operating
    N_ROT = 1
    N_COL = 13
    N_ROW = 4
    @classmethod
    def deinterleave(cls, input):
        cls.N_CBPSSI = len(input)
        s = int(np.max([cls.N_BPSCS/2, 1]))
        r = np.arange(0, cls.N_CBPSSI) 
        j = (r + ((2 * (cls.iss - 1)) % 3 + 3 * np.floor((cls.iss - 1) / 3)) * cls.N_ROT * cls.N_BPSCS) % cls.N_CBPSSI
        
        i = s * np.floor(j / s) + (j + np.floor(cls.N_COL * j / cls.N_CBPSSI)) % s
        k = cls.N_COL * i - (cls.N_CBPSSI - 1) * np.floor(i / cls.N_ROW)
        print(k)
        output = np.zeros(len(input), dtype=input.dtype)  # Initialize output array
        for index in range(len(input)):
            output[int(k[index])] = input[index]
        return output