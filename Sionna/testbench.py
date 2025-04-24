import os

if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

print("CUDA_VISIBLE_DEVICES: ", os.getenv("CUDA_VISIBLE_DEVICES"))

import sionna.phy
import sys
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

import components.sync



#mdic = loadmat("waveforms/preamble_BW20MHZ_SR160MHZ_degraded.mat")
mdic = loadmat("waveforms/rxSamplesData14_60M_WIFI.mat")
# #mdic = loadmat("waveforms/WIFI_AC_BW40_SS1_MCS3_BCC_Fs80M.mat")

samples1 = mdic["IQsamples1"][0]
# samples2 = mdic["IQsamples2"][0]
samplingRate = mdic["sampling rate"][0][0]

samples1 = tf.convert_to_tensor(samples1, dtype=tf.complex64)
samples1 = tf.reshape(samples1, (1, samples1.shape[0]))

#samples1 = tf.concat([samples1, samples1], axis=0)
print("samples1 shape: ", samples1.shape)

LSTFsync1 = components.sync.LSTFtimeSync(samplingRate=samplingRate)
index = LSTFsync1(samples1)
print(index)

index1 = tf.where(index[0] == 0)
index1 = tf.transpose(index1)[0]
index = tf.gather(index[1], index1)

fig, ax = plt.subplots(1, 1, figsize=(10, 6),sharex=True)
ax.plot(20 * tf.math.log(tf.abs(samples1[0])) / tf.math.log(10.0))
for i in index:
    ax.axvline(x=i, color='r', linestyle='--')
ax.grid(True, which="both", axis="both")
ax.set_title("LSTF Time Sync")
fig.show()

LSTFfield1 = components.sync.LSTFfield(samplingRate)
LSTFfieldSample = LSTFfield1(samples1[0], index)

LLTFfield1 = components.sync.LLTFfield(samplingRate)
LLTFfieldSample = LLTFfield1(samples1[0], index)

LSIGfield1 = components.sync.LSIGfield(samplingRate)
LSIGfieldSample = LSIGfield1(samples1[0], index)

fig,ax = plt.subplots(3, 1, figsize=(10, 6))
for k in range(LSTFfieldSample.shape[0]):
    ax[0].plot(20 * tf.math.log(tf.abs(LSTFfieldSample[k])) / tf.math.log(10.0))
for k in range(LLTFfieldSample.shape[0]):
    ax[1].plot(20 * tf.math.log(tf.abs(LLTFfieldSample[k])) / tf.math.log(10.0))
for k in range(LSIGfieldSample.shape[0]):
    ax[2].plot(20 * tf.math.log(tf.abs(LSIGfieldSample[k])) / tf.math.log(10.0))

ax[0].grid(True, which="both", axis="both")
ax[1].grid(True, which="both", axis="both")
ax[2].grid(True, which="both", axis="both")
ax[0].set_title("LSTF LLTF LSIG")
fig.show()

LSTFfreqSync1 = components.sync.LSTFfreqSync(samplingRate=samplingRate)
freqOffset = LSTFfreqSync1(LSTFfieldSample)
print(freqOffset)
LLTFfreqSync1 = components.sync.LLTFfreqSync(samplingRate=samplingRate)
freqOffset = LLTFfreqSync1(LLTFfieldSample)
print(freqOffset)

freqComp = components.sync.freqComp(samplingRate=samplingRate)
LLTFfieldSample = freqComp(LLTFfieldSample, freqOffset)

bw = components.sync.BW.BW_20MHZ
LLTFchannelEst = components.sync.LLTFchannelEst(samplingRate=samplingRate, bw=bw)
channel = LLTFchannelEst(LLTFfieldSample)

LSIGfieldSample = freqComp(LSIGfieldSample, freqOffset)
LSIGdemod = components.sync.LSIGdemodulator(samplingRate=samplingRate, bw=bw)
LSIGsymbol = LSIGdemod(LSIGfieldSample, channel)

LSIGdecoder = components.sync.LSIGdecoder()
LSIGresult = LSIGdecoder(LSIGsymbol)

fig, ax = plt.subplots(1, 1, figsize=(10, 6), sharex=True)
for k in range(LSIGresult.shape[0]):
    ax.plot(LSIGresult[k,:],'-o')
ax.grid(True, which="both", axis="both")
ax.set_title("LSIG bits")
fig.show()

# encoder = sionna.phy.fec.conv.ConvEncoder(['1011011','1111001'])
# decoder = sionna.phy.fec.conv.BCJRDecoder(encoder)
# data = tf.constant([1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=tf.int32)
# data = tf.expand_dims(data, axis=0)
# encoded = encoder(data)
# print("encoded: ", encoded.numpy())

input("Press Enter to continue...")