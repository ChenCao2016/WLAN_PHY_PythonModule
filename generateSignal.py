import numpy as np
from scipy.fft import fft, ifft, fftshift, fftfreq
import ChannelCoding
import Interleaving
import Constellation

class timeWindows:
    def __init__(self):
        #result
        self.coef = None

    def __call__(self, t,transitionTime,symbolTime):
        if t >= - transitionTime/2 and t < transitionTime/2:
            self.coef = np.sin(np.pi/2*(0.5+t/transitionTime))**2
        elif t >= transitionTime/2 and t < symbolTime - transitionTime/2:
            self.coef = 1
        elif t >= symbolTime - transitionTime/2 and t <= symbolTime + transitionTime/2:
            self.coef = np.sin(np.pi/2*(0.5-(t-symbolTime)/transitionTime))**2
        else:
            raise Exception(f"Time {t} out of range: {-transitionTime/2} to {symbolTime + transitionTime/2}")
        
        return
timeWindows = timeWindows()

class generateLSTF:
    def __init__(self):
        self.LSTF_duration = 8e-6
        self.k = np.arange(-26,27,1)
        self.S26 = np.sqrt(1/2)*np.array([
            0, 0, 1 + 1j, 0, 0, 0, -1 - 1j, 0, 0, 0, 1 + 1j, 0, 0, 0, -1 - 1j, 0, 0, 0, -1 - 1j, 0, 0, 0, 1 + 1j, 0, 0, 0,
            0, 0, 0, 0, -1 - 1j, 0, 0, 0, -1 - 1j, 0, 0, 0, 1 + 1j, 0, 0, 0, 1 + 1j, 0, 0, 0, 1 + 1j, 0, 0, 0, 1 + 1j, 0, 0,
        ])
        self.subcarrierSpace = 312.5e3
        self.Ntone = 6*2
        self.Ntx = 1

        #result
        self.LSTF_timeDomain = None
        self.startIndex = None
        self.endIndex = None
    
    def __call__(self, BW, samplingRate, transitionTime = 0):
        if BW == 20e6:
            self.LSTF_timeDomain = np.zeros(int((self.LSTF_duration+transitionTime)*samplingRate),dtype=complex)
            startIndex = -int(transitionTime/2*samplingRate)
            for n in range(startIndex,startIndex+len(self.LSTF_timeDomain)):
                t = n/samplingRate
                timeWindows(t,transitionTime,self.LSTF_duration)
                self.LSTF_timeDomain[n-startIndex] = 1/np.sqrt(self.Ntone*self.Ntx)*timeWindows.coef*np.sum(self.S26*np.exp(1j*2*np.pi*self.k*self.subcarrierSpace*t))
            self.startIndex = -startIndex
            self.endIndex = startIndex+len(self.LSTF_timeDomain)
        else:
            self.LSTF_timeDomain = None
            self.startIndex = None
            self.endIndex = None
            raise Exception("BW not supported")
        return
generateLSTF = generateLSTF()

class generateLLTF:
    def __init__(self):
        self.LLTF_duration = 8e-6
        self.k = np.arange(-26,27,1)
        self.L26 = np.array([
            1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 0,
            1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1,            
        ])
        self.subcarrierSpace = 312.5e3
        self.TGI2 = 1.6e-6
        self.Ntone = 26*2
        self.Ntx = 1

        #result
        self.LLTF_timeDomain = None
        self.startIndex = None
        self.endIndex = None
    def __call__(self, BW, samplingRate, transitionTime = 0):
        if BW == 20e6:
            self.LLTF_timeDomain = np.zeros(int(self.LLTF_duration*samplingRate),dtype=complex)
            startIndex = -int(transitionTime/2*samplingRate)
            for n in range(startIndex,startIndex+len(self.LLTF_timeDomain)):
                t = n/samplingRate
                timeWindows(t,transitionTime,self.LLTF_duration)
                self.LLTF_timeDomain[n-startIndex] = 1/np.sqrt(self.Ntone*self.Ntx)*timeWindows.coef*np.sum(self.L26*np.exp(1j*2*np.pi*self.k*self.subcarrierSpace*(t - self.TGI2)))
            self.startIndex = -startIndex
            self.endIndex = startIndex+len(self.LLTF_timeDomain)
        else:
            self.LLTF_timeDomain = None
            self.startIndex = None
            raise Exception("BW not supported")
        return
generateLLTF = generateLLTF()

class generateLSTFandLLTF:
    def __init__(self):
        self.result = None
        self.startIndex = None 
        self.endIndex = None
        self.midIndex = None

    def __call__(self, BW, samplingRate, transitionTime = 0):
        generateLSTF(BW, samplingRate, transitionTime)
        generateLLTF(BW, samplingRate, transitionTime)

        lstf = generateLSTF.LSTF_timeDomain
        lltf = generateLLTF.LLTF_timeDomain
        lstf[-generateLLTF.startIndex:] = lstf[-generateLLTF.startIndex:] + lltf[:generateLLTF.startIndex]
        self.result = np.concatenate((lstf,lltf[generateLLTF.startIndex:]))
        self.startIndex = generateLSTF.startIndex
        self.endIndex = len(self.result) - generateLSTF.startIndex
        self.midIndex = generateLSTF.endIndex

        return
generateLSTFandLLTF = generateLSTFandLLTF()


class generateLSIG:
    def __init__(self):
        self.LSIG_duration = 4e-6
        self.TGI = 0.8e-6
        self.k = np.arange(-26,27,1)
        self.subcarrierSpace = 312.5e3
        self.Ntone = 26*2
        self.Ntx = 1
        self.pliot64 = [11, 25, 39, 53]
        self.pliot64Symbol = [1, 1, 1, -1]
        self.DC64 = 32
        self.edge = [0,1,2,3,4,5, 63,62,61,60,59]

        #result
        self.LSIG_timeDomain = None
        self.startIndex = None
        self.endIndex = None

    def __call__(self, content, BW, samplingRate, transitionTime = 0):
        if len(content) != 24:
            raise Exception(f"LSIG bits should be 24, but got {len(content)}")
        
        encoded = ChannelCoding.BCCEncoder(content, "1/2")

        Interleaving.interleaver(encoded, 1)
        interleaved = Interleaving.interleaver.forward()
        symbols = Constellation.BPSKmapper(interleaved)

        if BW == 20e6:
            LSIG_subcarrier = np.zeros(64,dtype=complex)
            symbolCounter = 0
            for i in range(64):
                if i in self.pliot64:
                    LSIG_subcarrier[i] = self.pliot64Symbol[self.pliot64.index(i)]
                elif i == self.DC64 or i in self.edge:
                    LSIG_subcarrier[i] = 0
                else:
                    LSIG_subcarrier[i] = symbols[symbolCounter]
                    symbolCounter += 1
                    

            self.LSIG_timeDomain = np.zeros(int(self.LSIG_duration*samplingRate),dtype=complex)
            startIndex = -int(transitionTime/2*samplingRate)
            for n in range(startIndex,startIndex+len(self.LSIG_timeDomain)):
                t = n/samplingRate
                timeWindows(t,transitionTime,self.LSIG_duration)
                self.LSIG_timeDomain[n-startIndex] = 1/np.sqrt(self.Ntone*self.Ntx)*timeWindows.coef*np.sum(LSIG_subcarrier[6:-5]*np.exp(1j*2*np.pi*self.k*self.subcarrierSpace*(t-self.TGI)))
            self.startIndex = -startIndex
            self.endIndex = startIndex+len(self.LSIG_timeDomain)
        else:
            self.LSIG_timeDomain = None
            self.startIndex = None
            self.endIndex = None
            raise Exception("BW not supported")
        return
    
generateLSIG = generateLSIG()

class generateLegacyPreamble:
    def __init__(self):
        self.result = None
        self.startIndex = None 
        self.endIndex = None
        self.midIndex1 = None
        self.midIndex2 = None

    def __call__(self, BW, samplingRate, SIGbit, transitionTime = 0):
        generateLSTFandLLTF(BW, samplingRate, transitionTime)
        generateLSIG(SIGbit, BW, samplingRate, transitionTime)

        ltf = generateLSTFandLLTF.result
        lsig = generateLSIG.LSIG_timeDomain
        ltf[-generateLSIG.startIndex:] = ltf[-generateLSIG.startIndex:] + lsig[:generateLSIG.startIndex]
        self.result = np.concatenate((ltf,lsig[generateLSIG.startIndex:]))
        self.startIndex = generateLSTFandLLTF.startIndex
        self.endIndex = len(self.result) - generateLSIG.startIndex
        self.midIndex1 = generateLSTFandLLTF.midIndex
        self.midIndex2 = generateLSTFandLLTF.endIndex

generateLegacyPreamble = generateLegacyPreamble()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    samplingRate = 160e6
    BW = 20e6
    generateLSTF(BW, samplingRate, 100e-9)
    sstf = generateLSTF.LSTF_timeDomain

    fig, ax = plt.subplots()
    ax.plot(np.abs(sstf))
    ax.plot(generateLSTF.startIndex, np.abs(sstf)[generateLSTF.startIndex],'o')
    ax.plot(generateLSTF.endIndex, np.abs(sstf)[generateLSTF.endIndex],'o')

    fig.suptitle("Time domain signal")
    fig.show()

    fig, ax = plt.subplots()
    ax.plot(fftshift(np.abs(fft(sstf))))
    fig.suptitle("Freq domain signal")
    fig.show()


    generateLLTF(BW, samplingRate, 100e-9)
    lltf = generateLLTF.LLTF_timeDomain

    fig, ax = plt.subplots()
    ax.plot(np.abs(lltf))
    ax.plot(generateLLTF.startIndex, np.abs(lltf)[generateLLTF.startIndex],'o')
    ax.plot(generateLLTF.endIndex, np.abs(lltf)[generateLLTF.endIndex],'o')
    fig.suptitle("Time domain signal")
    fig.show()

    fig, ax = plt.subplots()
    ax.plot(fftshift(np.abs(fft(lltf))))
    fig.suptitle("Freq domain signal")
    fig.show()

    LSIG_bits = [1,1,0,1,  0,  1,1,0,1,0,1,1,0,1,0,1,0,  0,  0,0,0,0,0,0]
    print(f"LSIG bits: {len(LSIG_bits)} , {LSIG_bits}")
    generateLSIG(LSIG_bits, BW, samplingRate, 100e-9)
    lsig = generateLSIG.LSIG_timeDomain

    fig, ax = plt.subplots()
    ax.plot(np.abs(lsig))
    ax.plot(generateLSIG.startIndex, np.abs(lsig)[generateLSIG.startIndex],'o')
    ax.plot(generateLSIG.endIndex, np.abs(lsig)[generateLSIG.endIndex],'o')
    fig.suptitle("Time domain signal")
    fig.show()

    generateLegacyPreamble(BW, samplingRate, LSIG_bits, 100e-9)
    preamble = generateLegacyPreamble.result
    preamble = 10*np.log10(np.abs(preamble)**2)
    fig, ax = plt.subplots()
    ax.plot(preamble)
    ax.plot(generateLegacyPreamble.startIndex, preamble[generateLegacyPreamble.startIndex],'o', color="tab:orange")
    ax.plot(generateLegacyPreamble.midIndex1, preamble[generateLegacyPreamble.midIndex1],'o', color="tab:orange")
    ax.plot(generateLegacyPreamble.midIndex2, preamble[generateLegacyPreamble.midIndex2],'o', color="tab:orange")
    ax.plot(generateLegacyPreamble.endIndex, preamble[generateLegacyPreamble.endIndex],'o', color="tab:orange")
    fig.show()

    # generateLSTFandLLTF(BW, samplingRate, 100e-9)
    # preamble = generateLSTFandLLTF.result
    # preamble = 10*np.log10(np.abs(preamble)**2)
    # fig, ax = plt.subplots()
    # ax.plot(preamble)
    # ax.plot(generateLSTFandLLTF.startIndex, preamble[generateLSTFandLLTF.startIndex],'o')
    # ax.plot(generateLSTFandLLTF.endIndex, preamble[generateLSTFandLLTF.endIndex],'o')
    # ax.plot(generateLSTFandLLTF.midIndex, preamble[generateLSTFandLLTF.midIndex],'o')
    # fig.suptitle("Time domain signal")
    # fig.show()

    input("Press Enter to continue...")