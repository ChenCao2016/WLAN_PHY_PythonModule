
#This is the script for checking IQimbalance, TAL-1005
from matplotlib import pyplot

from scipy.io import loadmat
from scipy.fft import fft, ifft, fftshift, fftfreq
from scipy.signal import istft, stft
import numpy as np

from component.Synchronization import *
from component.LSIGdecoder import *
from component.USIGdecoder import *
from component.ChannelCoding import *
from component.Constellation import *
from component.IQimbalance import *

#---------------------------------------------------------------
#mdic = loadmat("waveforms/Verify_2412_SaveAlways-128.2025.2-28.17.17.10_PSDU=448_ON.mat")

#mdic = loadmat("waveforms/Verify_2412_SaveAlways-128.2025.2-28.17.18.50_PSDU=428_ON.mat")
#mdic = loadmat("waveforms/rxSamplesData15_80M_WIFI.mat")
mdic = loadmat("waveforms/rxSamplesData13_100M_WIFI.mat")

samples1 = mdic["IQsamples1"][0]
# samples2 = mdic["IQsamples2"][0]
samplingRate = mdic["sampling rate"][0][0]
print("sampling rate: {}".format(samplingRate))

fig,ax = pyplot.subplots()
ax.plot(20*np.log10(np.abs(samples1)))
# ax.plot(20*np.log10(np.abs(samples2)))
ax.set_title('Time domain')
ax.grid()
fig.show()

signalPST1 = np.array(mdic["pst1"][0])
# signalPST2 = np.array(mdic["pst2"][0])

# signalPST1 = signalPST1[1::2]
# signalPST2 = signalPST2[1::2]

# print("signalPST1: {}".format(signalPST1))
# print("signalPST2: {}".format(signalPST2))

signal1PSTindex = (signalPST1*samplingRate).astype(int)
# signal2PSTindex = (signalPST2*samplingRate).astype(int)

print("signal1PSTindex: {}".format(signal1PSTindex))
# print("signal2PSTindex: {}".format(signal2PSTindex))

# #---------------------------------------------------------------
# # find the end of L-STF
# LSTF_endIndex = []
# LSTF_sync(samples, samplingRate, LSTF_endIndex)
# LSTF_endIndex[0] = LSTF_endIndex[0] - 1

# # LSTF_endIndex[0] = 1280 + 160

LSTF_endIndex = signal1PSTindex+int(8e-6*samplingRate) - 10
packetIndex = 0
# #---------------------------------------------------------------
# # L-STF, L-LTF, L-SIG
print(f"Packet end index: {LSTF_endIndex[packetIndex]}")
print(f"Packet end time: {LSTF_endIndex[packetIndex]/samplingRate}s")

LSTF_endTime = LSTF_endIndex[packetIndex]/samplingRate
LSTF_startTime = LSTF_endTime - 8e-6
LSTF_startIndex = int(LSTF_startTime*samplingRate)

LLTF_startIndex1 = LSTF_endIndex[packetIndex] + int(1.6e-6*samplingRate)
LLTF_endIndex1 = LLTF_startIndex1 + int(3.2e-6*samplingRate)
LLTF_startIndex2 = LLTF_endIndex1
LLTF_endIndex2 = LLTF_startIndex2 + int(3.2e-6*samplingRate)

LSIG_startIndex = LLTF_endIndex2 + int(0.8e-6*samplingRate)
LSIG_endIndex = LSIG_startIndex + int(3.2e-6*samplingRate)


# f, t_stft, Zxx = stft(samples1[:LSIG_endIndex], samplingRate, nperseg=100,return_onesided=False)
# Zxx = 20*np.log10(np.abs(Zxx))
# fig = pyplot.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')
# T, F = np.meshgrid(t_stft, f)
# Z = Zxx
# surf = ax.plot_surface(T, F, Z, cmap='viridis', edgecolor='none')
# ax.set_xlabel('Time (s)')
# ax.set_ylabel('Frequency e8 (Hz)')
# ax.set_zlabel('Magnitude')
# fig.colorbar(surf, ax=ax, label='Magnitude')
# fig.show()

#---------------------------------------------------------------
# Estimate and compensate fractional frequency offset
freqOffset1 = []
LSTF_freqOffset(samples1, samplingRate, LSTF_endIndex[packetIndex], freqOffset1)
print(f"Frequency offset from LSTF: {freqOffset1[0]}Hz")

freqOffset2 = []
LLTF_freqOffset(samples1, samplingRate, LSTF_endIndex[packetIndex], freqOffset2)
print(f"Frequency offset from LLTF: {freqOffset2[0]}Hz")

newSamples = []
freqCompensate(samples1, samplingRate, freqOffset1[0], newSamples)

#---------------------------------------------------------------
# Display L-STF, L-LTF, L-SIG
fig, (ax1,ax2) = pyplot.subplots(2,1)
ax1.plot(np.abs(newSamples[LSTF_startIndex:LSTF_endIndex[packetIndex]]))
ax1.set_title('LSTF time domain')
ax2.plot(fftshift(fftfreq(len(newSamples[LSTF_startIndex:LSTF_endIndex[packetIndex]])))*samplingRate,np.abs(fftshift(fft(newSamples[LSTF_startIndex:LSTF_endIndex[packetIndex]]))))
ax2.set_title('LSTF frequency domain')
fig.tight_layout()
fig.show()


fig, ((ax1,ax2),(ax3,ax4)) = pyplot.subplots(2,2)
ax1.plot(np.abs(newSamples[LLTF_startIndex1:LLTF_endIndex1]))
ax1.plot(np.abs(newSamples[LLTF_startIndex2:LLTF_endIndex2]))
ax1.set_title('LLTF time domain')
ax2.plot(fftshift(fftfreq(len(newSamples[LLTF_startIndex1:LLTF_endIndex1])))*samplingRate,np.abs(fftshift(fft(newSamples[LLTF_startIndex1:LLTF_endIndex1]))),marker='.',label='LLTF1')
ax2.plot(fftshift(fftfreq(len(newSamples[LLTF_startIndex2:LLTF_endIndex2])))*samplingRate,np.abs(fftshift(fft(newSamples[LLTF_startIndex2:LLTF_endIndex2]))))
ax2.set_title('LLTF frequency domain')
ax3.plot(np.unwrap(np.angle(newSamples[LLTF_startIndex1:LLTF_endIndex1])))
ax3.plot(np.unwrap(np.angle(newSamples[LLTF_startIndex2:LLTF_endIndex2])))
ax3.grid()
ax3.set_title('LLTF time domain')
ax4.plot(fftshift(fftfreq(len(newSamples[LLTF_startIndex1:LLTF_endIndex1])))*samplingRate,np.angle(fftshift(fft(newSamples[LLTF_startIndex1:LLTF_endIndex1]))))
ax4.plot(fftshift(fftfreq(len(newSamples[LLTF_startIndex2:LLTF_endIndex2])))*samplingRate,np.angle(fftshift(fft(newSamples[LLTF_startIndex2:LLTF_endIndex2]))))
ax4.set_title('LLTF frequency domain')
fig.tight_layout()
fig.show()
fig, (ax1,ax2) = pyplot.subplots(2,1)
ax1.plot(np.abs(newSamples[LSIG_startIndex:LSIG_endIndex]))
ax1.set_title('LSIG time domain')
ax2.plot(fftshift(fftfreq(len(newSamples[LSIG_startIndex:LSIG_endIndex])))*samplingRate,np.abs(fftshift(fft(newSamples[LSIG_startIndex:LSIG_endIndex]))))
ax2.set_title('LSIG frequency domain')
fig.tight_layout()
fig.show()

#----------------------------------------------------------------------------------------
# Estimate channel based on L-LTF
LLTF_channel = []
LLTF_channelEstimate(newSamples, samplingRate, LSTF_endIndex[packetIndex], LLTF_channel)

# LLTF is only 26 + 26 subcarriers (including 4 pilot subcarriers) in 20MHz bandwidth

channelPhase = []
channelAmplitude = []

for i in LLTF_channel:
    channelPhase.append(cmath.phase(i))
    channelAmplitude.append(abs(i))

fig, ax = pyplot.subplots()
ax.plot(channelAmplitude)
ax.set_xlabel('Sample Index')
ax.set_ylabel('Amplitude')
ax.set_title('Channel Amplitude')
fig.show()

fig, ax = pyplot.subplots()
ax.plot(channelPhase)
ax.set_xlabel('Sample Index')
ax.set_ylabel('Phase')
ax.set_title('Channle Phase')
fig.show()

#----------------------------------------------------------------------------------------
#estimate IQ imbalance
IQimbalance_LLTF(newSamples, samplingRate, LSTF_endIndex[packetIndex], LLTF_channel)


#----------------------------------------------------------------------------------------
# Decode L-SIG
LSIG_symbol = []
LSIG_demodulator(newSamples, samplingRate,
                 LSTF_endIndex[packetIndex], LLTF_channel, LSIG_symbol)


fig, ax = pyplot.subplots()
ax.scatter(np.real(LSIG_symbol), np.imag(LSIG_symbol))
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel('LSIG_symbol Real Part')
ax.set_ylabel('LSIG_symbol Imaginary Part')
ax.set_title('Scatter Plot of LSIG_symbol')
fig.show()

evmp = BPSK_EVM(LSIG_symbol)
print(f"EVM % of LSIG: {evmp*100}%")
print(f"EVM dB of LSIG: {20*np.log10(evmp)}dB")

#For HT and VHT formats, the L-SIG rate bits are set to '1 1 0 1'. Data rate information for HT and VHT formats is signaled in format-specific signaling fields.
LSIG_bits = []
LSIG_info = {}
LSIG_decoder(LSIG_symbol, LSIG_bits, LSIG_info)
print("LSIG information:")
print(LSIG_info)

input("I will stop at L-SIG decoding")
