import cmath


# ----------------------------------------------------------------------
# Discrete Fourier Transform
# Number of DFT points is determined by the length of input
# ----------------------------------------------------------------------
# INPUT
# timeDomainSamples: complex number array
# ----------------------------------------------------------------------
# OUTPUT
# frequencyDomainSamples: complex number array
# ----------------------------------------------------------------------
def DFT(timeDomainSamples, frequencyDomainSamples):

    frequencyDomainSamples.clear()
    point = len(timeDomainSamples)

    for x in range(point):
        temp = 0
        for y in range(point):
            temp = temp + timeDomainSamples[y] * \
                cmath.exp(-1j*2*cmath.pi*x*y/point)
        frequencyDomainSamples.append(temp)
    return


# ----------------------------------------------------------------------
# Inverse Discrete Fourier Transform
# Number of points is determined by the length of input
# ----------------------------------------------------------------------
# INPUT
# frequencyDomainSamples: complex number array
# ----------------------------------------------------------------------
# OUTPUT
# timeDomainSamples: complex number array
# ----------------------------------------------------------------------
def IDFT(frequencyDomainSamples, timeDomainSamples):

    timeDomainSamples.clear()
    point = len(frequencyDomainSamples)

    for x in range(point):
        temp = 0
        for y in range(point):
            temp = temp + timeDomainSamples[y] * \
                cmath.exp(1j*2*cmath.pi*x*y/point)
        frequencyDomainSamples.append(temp)

    return
