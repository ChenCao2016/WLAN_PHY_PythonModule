from .GlobeParameter import *
from .DFT import *
from .Interleaving import *
from .Constellation import *
from .ChannelCoding import *
from scipy.fft import fft, ifft, fftshift, fftfreq
import numpy as np



# ----------------------------------------------------------------------
# Legacy SIG field demodulation
# ----------------------------------------------------------------------
#INPUT#
# samples: complex number array
#samplingRate: number (Hz)
# LSTF_endIndex: sample index for LSTF end
# LLTF_channel: channel estimation on 64 subcarrier 20MHz bandwidth
# ----------------------------------------------------------------------
#OUTPUT#
# LSIG_symbol: legacy SIG field constellation symbols array
# ----------------------------------------------------------------------
def USIG_demodulator(samples, samplingRate, USIG_startIndex, LLTF_channel, LSIG_symbol):
    ret = 0
    try:

        startIndex = USIG_startIndex + 1
        ratio = int(samplingRate/L_preambleBandth)
        windowSize = int((LSIG_length - L_cp) * samplingRate)
        
        # we will need index shift for bandwith beyond 20MHz
        # temp = []
        # DFT(samples[startIndex:startIndex + windowSize: ratio], temp)
        temp = fftshift(fft(samples[startIndex:startIndex + windowSize: ratio]))

        # Phase track
        phaseShift = 0
        for i in pilot64.index:
            phaseShift = phaseShift + \
                (temp[i]/LLTF_channel[i]) / \
                pilot64.symbol[pilot64.index.index(i)]
        phaseShift = phaseShift/pilot64.num
        if abs(phaseShift) < 1e-6:
            phaseShift = 1
        #phaseShift = 1

        # Phase track
        amplitudeShift = 0
        for i in pilot64.index:
            amplitudeShift = amplitudeShift + \
                np.abs(temp[i]/LLTF_channel[i]) / \
                np.abs(pilot64.symbol[pilot64.index.index(i)])
        amplitudeShift = amplitudeShift/pilot64.num


        symbol = []
        for i in range(len(LLTF_channel)):
            if i not in pilot64.index:
                if np.real(LLTF_channel[i]) == 0:
                    # symbol.append(0);
                    pass
                else:
                    symbol.append(temp[i]/LLTF_channel[i]/phaseShift)
        # -----------------------------------------------------------------------------------------------------------
        LSIG_symbol.clear()
        for x in symbol:
            LSIG_symbol.append(x)

    except Exception as err:
        ret = -1
        print(str(err))
    return ret


# ----------------------------------------------------------------------
# Legacy SIG field decoder
# ----------------------------------------------------------------------
#INPUT#
# LSIG_symbol: legacy SIG field constellation symbols array
# ----------------------------------------------------------------------
#OUTPUT#
# LSIG_bits: legacy SIG field bits array
# LSIG_info: legacy SIG field information dictionary
# ----------------------------------------------------------------------
def USIG_decoder(LSIG1_symbol, LSIG2_symbol, LSIG_bits):
    ret = 0
    LLR1 = []
    #BPSKDemapperCenter(LSIG1_symbol, LLR1, center = (-1,1),variance=10)

    BPSKDemapper(LSIG1_symbol, LLR1, variance=1)

    LLR2 = []
    BPSKDemapper(LSIG2_symbol, LLR2, variance=1)


    BCCDeinterleaver.N_COL = 13
    BCCDeinterleaver.N_ROW = 4

    dLLR1 = BCCDeinterleaver.deinterleave(np.array(LLR1))
    dLLR2 = BCCDeinterleaver.deinterleave(np.array(LLR2))

    #dLLR1 = np.array(dLLR1)
    #dLLR1[np.abs(dLLR1) < 1.5] = 0
    # dLLR2 = np.array(dLLR2)
    # dLLR2 =  dLLR2*200

    dLLR = np.concatenate((np.array(dLLR1), np.array(dLLR2)))
    print(dLLR)

    # dLLR_bits = []
    # for x in dLLR:
    #     if x > 0:
    #         dLLR_bits.append(0)
    #     else:
    #         dLLR_bits.append(1)

    # print(dLLR_bits)


    decoded_LLR = BCCDecoder(dLLR, "1/2")

    print(decoded_LLR)
    LSIG_bits.clear()
    for x in decoded_LLR:
        if x >= 0:
            LSIG_bits.append(0)
        else:
            LSIG_bits.append(1)

    return ret


def CRC_calc(input):

    c = [1,1,1,1,1,1,1,1]

    for i in range(len(input)):
        temp = input[i]^c[7]
        c[7] = c[6]
        c[6] = c[5]
        c[5] = c[4]
        c[4] = c[3]
        c[3] = c[2]
        c[2] = c[1]^temp
        c[1] = c[0]^temp
        c[0] = temp

    for i in range(len(c)):
        c[i] = 1 - c[i]

    return [c[7], c[6], c[5], c[4]]





