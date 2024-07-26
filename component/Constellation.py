import numpy as np

def BPSKDemapper(symbols, LLRs, variance=1):
    ret = 0

    LLRs.clear()

    for x in symbols:
        LLRs.append((((x.real - 1)**2 + x.imag**2) -
                    ((x.real + 1)**2 + x.imag**2))/2/variance)

    return ret



def BPSKDemapperCenter(symbols, LLRs, center = (-1,1), variance=1):
    ret = 0

    LLRs.clear()

    for x in symbols:
        LLRs.append((((x.real - center[1])**2 ) -
                    ((x.real - center[0])**2 ))/2/variance)

    return ret


def BPSKmapper(bits):

    symbols = []

    for x in bits:
        if x == 0:
            symbols.append(-1)
        elif x == 1:    
            symbols.append(1)
        else:
            raise Exception(f"Invalid bit {x}")

    return symbols



def BPSK_EVM(symbols):

    ref = np.zeros(len(symbols),dtype=complex)

    ref[np.real(symbols) > 0] = 1
    ref[np.real(symbols) < 0] = -1

    evm = np.sqrt(np.mean(np.real(symbols - ref)**2 + np.imag(symbols - ref)**2))

    return evm