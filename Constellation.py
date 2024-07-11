
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
