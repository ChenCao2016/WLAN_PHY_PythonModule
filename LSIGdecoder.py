from GlobeParameter import *
from DFT import *
from Interleaving import *
from Constellation import *
from ChannelCoding import *
import math
import cmath


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
def LSIG_demodulator(samples, samplingRate, LSTF_endIndex, LLTF_channel, LSIG_symbol):
    ret = 0
    try:

        startIndex = LSTF_endIndex + int(samplingRate*(LLTF_length + L_cp)) + 1
        ratio = int(samplingRate/L_preambleBandth)
        windowSize = int((LSIG_length - L_cp) * samplingRate)
        temp = []
        DFT(samples[startIndex:startIndex + windowSize: ratio], temp)

        # Phase track
        phaseShift = 0
        for i in pilot64.index:
            phaseShift = phaseShift + \
                (temp[i]/LLTF_channel[i]) / \
                pilot64.symbol[pilot64.index.index(i)]
        phaseShift = phaseShift/pilot64.num
        if abs(phaseShift) < 1e-6:
            phaseShift = 1

        symbol = []
        for i in range(len(LLTF_channel)):
            if i not in pilot64.index:
                if LLTF_channel[i] == 0:
                    # symbol.append(0);
                    pass
                else:
                    symbol.append(temp[i]/LLTF_channel[i]/phaseShift)

        # -----------------------------------------------------------------------------------------------------------
        # RL-SIG detection
        startIndex2 = LSTF_endIndex + \
            int(samplingRate*(LLTF_length + L_cp + LSIG_length)) + 1
        temp = []
        DFT(samples[startIndex2:startIndex2 + windowSize: ratio], temp)

        # Phase track
        phaseShift = 0
        for i in pilot64.index:
            phaseShift = phaseShift + \
                (temp[i]/LLTF_channel[i]) / \
                pilot64.symbol[pilot64.index.index(i)]
        phaseShift = phaseShift/pilot64.num
        if abs(phaseShift) < 1e-6:
            phaseShift = 1

        symbol2 = []
        for i in range(len(LLTF_channel)):
            if i not in pilot64.index:
                if LLTF_channel[i] == 0:
                    # symbol.append(0)
                    pass
                else:
                    symbol2.append(temp[i]/LLTF_channel[i]/phaseShift)

        correlation = 0
        power = 0
        for i in range(len(symbol)):
            correlation += symbol[i]*conj(symbol2[i])
            power += abs(symbol[i])*abs(symbol2[i])

        if abs(correlation)/power > 0.9:
            # This is 11AX
            # MRC
            for i in range(len(symbol)):
                symbol[i] = (symbol[i]/abs(symbol[i]) +
                             symbol2[i]/abs(symbol2[i]))/2

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
def LSIG_decoder(LSIG_symbol, LSIG_bits, LSIG_info):
    ret = 0

    try:
        LLR = []
        BPSKDemapper(LSIG_symbol, LLR)

        dLLR = []
        Deinterleaver(LLR, 1, dLLR)

        decoded_LLR = []
        BCCDecoder(dLLR, "1/2", decoded_LLR)

        LSIG_bits.clear()
        for x in decoded_LLR:
            if x > 0:
                LSIG_bits.append(0)
            else:
                LSIG_bits.append(1)

        parity = 0
        for i in range(17):
            parity = parity ^ LSIG_bits[i]

        if parity == LSIG_bits[17]:
            LSIG_info["Parity"] = "PASS"
        else:
            LSIG_info["Parity"] = "FAIL"

        LSIG_info["Rate"] = str(
            LSIG_bits[0]) + str(LSIG_bits[1]) + str(LSIG_bits[2]) + str(LSIG_bits[3])

        length = 0
        for i in range(12):
            length = length + LSIG_bits[5+i]*(2**i)

        LSIG_info["Length"] = length

    except Exception as err:
        ret = -1
        print(str(err))

    return ret
