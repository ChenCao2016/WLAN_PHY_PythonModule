from .GlobeParameter import *
from .DFT import *
from .Interleaving import *
from .Constellation import *
from .ChannelCoding import *
from scipy.fft import fft, ifft, fftshift, fftfreq

from matplotlib import pyplot


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
        
        # we will need index shift for bandwith beyond 20MHz
        # temp = []
        # DFT(samples[startIndex:startIndex + windowSize: ratio], temp)
        temp = fftshift(fft(samples[startIndex:startIndex + windowSize: ratio]))

        # Phase track
        # phaseShift = 0
        # for i in pilot64.index:
        #     phaseShift = phaseShift + \
        #         (temp[i]/LLTF_channel[i]) / \
        #         pilot64.symbol[pilot64.index.index(i)]
        # phaseShift = phaseShift/pilot64.num


        # Phase and amplitude track
        pilotShift = []
        for i in pilot64.index:
            pilotShift.append((temp[i]/LLTF_channel[i]) / pilot64.symbol[pilot64.index.index(i)])
        pilotShift = np.array(pilotShift)

        #pilotShift[:int(len(pilotShift)/2)] = np.conjugate(pilotShift[:int(len(pilotShift)/2)])

        # fig, (ax1,ax2) = pyplot.subplots(2,1)
        # ax1.plot(np.angle(pilotShift))
        # ax1.set_xlabel('Sample Index')
        # ax1.set_ylabel('channel phase')
        # ax1.set_title('channel phase')
        # ax2.plot(np.abs(pilotShift))
        # ax2.set_xlabel('Sample Index')
        # ax2.set_ylabel('channel amplitude')
        # ax2.set_title('channel amplitude')
        # fig.show()

        pilotShiftmean = np.mean(pilotShift)

        symbol = []
        for i in range(len(LLTF_channel)):
            if i not in pilot64.index:
                if LLTF_channel[i] == 0:
                    # symbol.append(0);
                    pass
                else:
                    symbol.append(temp[i]/LLTF_channel[i]/pilotShiftmean)

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

        #print(f"RL-SIG detection: {abs(correlation)/power}")
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

        decoded_LLR = BCCDecoder(dLLR, "1/2")

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
