import math


def Interleaver(input, N_CBPS, output):
    ret = 0

    output.clear()

    s = max(int(N_CBPS/2), 1)
    size = len(input)

    for j in range(size):

        i = s * floor(j/s) + (j + floor(16*j/size)) % s
        k = 16 * i - (size - 1)*floor(16*i/size)

        output.append(input[k])

    return


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
