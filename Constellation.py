
def BPSKDemapper(symbols, LLRs, variance = 1):
    ret = 0;

    LLRs.clear();

    for x in symbols:
          LLRs.append( (((x.real - 1)**2 + x.imag**2) - ((x.real + 1)**2 + x.imag**2))/2/variance );

    return ret;