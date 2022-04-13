import array
import math
from matplotlib import pyplot
from Synchronization import *
from LSIGdecoder import *

file = open("11AX.dat", "rb")

IQdata = array.array("f")
IQdata.fromstring(file.read())
file.close()

Idata = []
Qdata = []
for i in range(0, len(IQdata), 2):
    Idata.append(IQdata[i])
for i in range(1, len(IQdata), 2):
    Qdata.append(IQdata[i])

power = []
samples = []
for i in range(0, len(Idata)):
    samples.append(Idata[i] + 1j*Qdata[i])
    power.append(Idata[i]*Idata[i] + Qdata[i]*Qdata[i])

samplingRate = 80e6
LSTF_endIndex = []
LSTF_sync(samples, samplingRate, LSTF_endIndex)

print("Packet start index: " + str(LSTF_endIndex[0]))

freqOffset1 = []

LSTF_freqOffset(samples, samplingRate, LSTF_endIndex[0], freqOffset1)

print("Frequency offset: " + str(freqOffset1[0]) + "Hz")

freqOffset2 = []

LLTF_freqOffset(samples, samplingRate, LSTF_endIndex[0], freqOffset2)

newSamples = []

freqCompensate(samples, samplingRate, freqOffset1[0], newSamples)

LLTF_channel = []

LLTF_channelEstimate(newSamples, samplingRate, LSTF_endIndex[0], LLTF_channel)

channelPhase = []
channelAmplitude = []

for i in LLTF_channel:
    channelPhase.append(cmath.phase(i))
    channelPhase.append(abs(i))
# pyplot.plot(channelPhase);
# pyplot.show();

LSIG_symbol = []
LSIG_demodulator(newSamples, samplingRate,
                 LSTF_endIndex[0], LLTF_channel, LSIG_symbol)


LSIG_bits = []
LSIG_info = {}
LSIG_decoder(LSIG_symbol, LSIG_bits, LSIG_info)
print("LSIG information:")
print(LSIG_info)
