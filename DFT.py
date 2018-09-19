import cmath


def DFT(timeDomainSamples,frequencyDomainSamples):

    frequencyDomainSamples.clear();
    point = len(timeDomainSamples);

    for x in range(point):
        temp = 0;
        for y in range(point):
            temp = temp + timeDomainSamples[y]*cmath.exp(-1j*2*cmath.pi*x*y/point);
        frequencyDomainSamples.append(temp);
    return;


def IDFT(frequencyDomainSamples,timeDomainSamples):

    timeDomainSamples.clear();
    point = len(frequencyDomainSamples);

    for x in range(point):
        temp = 0;
        for y in range(point):
            temp = temp + timeDomainSamples[y]*cmath.exp(1j*2*cmath.pi*x*y/point);
        frequencyDomainSamples.append(temp);

    return;