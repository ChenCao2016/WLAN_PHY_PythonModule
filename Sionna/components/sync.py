import tensorflow as tf
import sionna.phy
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

class BW(Enum):
    BW_20MHZ = 0
    BW_40MHZ = 1
    BW_80MHZ = 2
    BW_160MHZ = 3

class LSTFtimeSync(sionna.phy.Block):
    def __init__(self, samplingRate: float):
        super().__init__()
        self.LSTF_length = 8e-6
        self.LSTF_windowRepeat = 10
        self.samplingRate = samplingRate

    #sample1 = [batch size, IQ sample length]
    #@tf.function    
    def call(self, samples1:tf.Tensor) -> tf.Tensor:

        windowSize = int(self.LSTF_length / 2 * self.samplingRate)
        correlation = tf.math.conj(samples1[:,:-windowSize]) * samples1[:,windowSize:]
        correlationValue = sionna.phy.signal.convolve(correlation, tf.ones(windowSize,dtype=tf.complex64), padding='valid', axis=1)
        correlationScale = sionna.phy.signal.convolve(tf.abs(correlation), tf.ones(windowSize,dtype=tf.complex64), padding='valid',axis=1)
        value1 = tf.abs(correlationValue/correlationScale)

        windowSize = int(self.LSTF_length / 5 * self.samplingRate)
        correlation = tf.math.conj(samples1[:,:-windowSize]) * samples1[:,windowSize:]
        correlationValue = sionna.phy.signal.convolve(correlation, tf.ones(windowSize,dtype=tf.complex64), padding='valid', axis=1)
        correlationScale = sionna.phy.signal.convolve(tf.abs(correlation), tf.ones(windowSize,dtype=tf.complex64), padding='valid', axis=1)
        value2 = tf.abs(correlationValue/correlationScale)

        value3 = value1 * value2[:, 0:value1.shape[1]]
        value3 = tf.where(value3 < 0.98, 0.0, value3)

        # 5-points peak detection
        value3 = (
            tf.where((value3[:,5:-5] - value3[:,4:-6]) > 0, 1, 0) *
            tf.where((value3[:,5:-5] - value3[:,3:-7]) > 0, 1, 0) *
            tf.where((value3[:,5:-5] - value3[:,2:-8]) > 0, 1, 0) *
            tf.where((value3[:,5:-5] - value3[:,1:-9]) > 0, 1, 0) *
            tf.where((value3[:,5:-5] - value3[:,:-10]) > 0, 1, 0) *
            tf.where((value3[:,5:-5] - value3[:,6:-4]) > 0, 1, 0) *
            tf.where((value3[:,5:-5] - value3[:,7:-3]) > 0, 1, 0) *
            tf.where((value3[:,5:-5] - value3[:,8:-2]) > 0, 1, 0) *
            tf.where((value3[:,5:-5] - value3[:,9:-1]) > 0, 1, 0) *
            tf.where((value3[:,5:-5] - value3[:,10:]) > 0, 1, 0)
        )

        indices = tf.where(value3 > 0)
        indices = tf.transpose(indices)
        indices = tf.concat([indices[0:1, :], indices[1:2, :] + 5], axis=0)
        #indices = [[batch index], [detect index]]

        return indices
    
class LSTFfield(sionna.phy.Block):
    def __init__(self, samplingRate: float):
        super().__init__()
        self.LSTF_length = 8e-6
        self.LSTF_windowRepeat = 10
        self.samplingRate = samplingRate

    @tf.function
    def call(self, samples1:tf.Tensor, startIndex:tf.Tensor) -> tf.Tensor:
        if len(samples1.shape) == 1:
            samples1 = tf.expand_dims(samples1, axis=0)
        sample_indices = startIndex[:, tf.newaxis] + tf.cast(tf.range(int(self.LSTF_length * self.samplingRate)), startIndex.dtype)[tf.newaxis, :]

        LSTFfield = tf.gather(samples1, sample_indices, axis = 1)[0]

        return LSTFfield

class LLTFfield(sionna.phy.Block):
    def __init__(self, samplingRate: float):
        super().__init__()
        self.LSTF_length = 8e-6
        self.LLTF_length = 8e-6
        self.LLTF_windowRepeat = 2
        self.samplingRate = samplingRate

    @tf.function
    def call(self, samples1:tf.Tensor, startIndex:tf.Tensor) -> tf.Tensor:
        if len(samples1.shape) == 1:
            samples1 = tf.expand_dims(samples1, axis=0)
        sample_indices = startIndex[:, tf.newaxis] + tf.cast(tf.range(int(self.LSTF_length * self.samplingRate),int((self.LSTF_length+ self.LLTF_length)* self.samplingRate)), startIndex.dtype)[tf.newaxis, :]

        LLTFfield = tf.gather(samples1, sample_indices, axis = 1)[0]
        return LLTFfield

class LSTFfreqSync(sionna.phy.Block):
    def __init__(self, samplingRate: float):
        super().__init__()
        self.LSTF_length = 8e-6
        self.LSTF_windowRepeat = 10
        self.samplingRate = samplingRate

    #@tf.function
    def call(self, samples1:tf.Tensor) -> tf.Tensor:

        windowSize = int(self.LSTF_length / self.LSTF_windowRepeat * self.samplingRate)
        correlation = tf.math.conj(samples1[:,:-windowSize]) * samples1[:,windowSize:]
        print(correlation.shape)

        #correlation = correlation[5:]
        freqOffset = tf.math.angle(tf.math.reduce_mean(correlation,axis=1))*self.samplingRate/np.pi/2/windowSize

        fig, ax = plt.subplots(1, 1, figsize=(10, 6), sharex=True)
        for k in range(correlation.shape[0]):
            ax.plot(tf.math.angle(correlation[k,:]), label='Correlation Phase')
        ax.set_title("LSTF Correlation Phase")
        fig.show()

        return freqOffset
    
class LLTFfreqSync(sionna.phy.Block):
    def __init__(self, samplingRate: float):
        super().__init__()
        self.LLTF_length = 8e-6
        self.LLTF_windowRepeat = 2
        self.LLTF_cp = 1.6e-6
        self.samplingRate = samplingRate

    #@tf.function
    def call(self, samples1:tf.Tensor) -> tf.Tensor:

        windowSize = int((self.LLTF_length-self.LLTF_cp) / self.LLTF_windowRepeat * self.samplingRate)
        index = int(self.LLTF_cp * self.samplingRate)
        correlation = tf.math.conj(samples1[:,:-windowSize]) * samples1[:,windowSize:]

        freqOffset = tf.math.angle(tf.math.reduce_mean(correlation,axis=1))*self.samplingRate/np.pi/2/windowSize

        fig, ax = plt.subplots(1, 1, figsize=(10, 6), sharex=True)
        for k in range(correlation.shape[0]):
            ax.plot(tf.math.angle(correlation[k,:]))
        ax.set_title("LLTF Correlation Phase")
        fig.show()

        return freqOffset
    
class LLTFchannelEst(sionna.phy.Block):
    def __init__(self, samplingRate: float, bw: Enum):
        super().__init__()
        self.samplingRate = samplingRate
        self.LLTF_length = 8e-6
        self.LLTF_CP = 1.6e-6

        if bw == BW.BW_20MHZ:
            self.preambleBW = 20e6
            self.LLTF = tf.constant([0, 0, 0, 0, 0, 0, 
            1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1,
            0, 
            1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 
            0, 0, 0, 0, 0], dtype=tf.complex64)
        elif bw == BW.BW_40MHZ:
            self.preambleBW = 40e6
            self.LLTF = tf.constant([
            0, 0, 0, 0, 0, 0,
            1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 0,
            1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 0,
            1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1,
            0, 0, 0, 0, 0,        
            ], dtype=tf.complex64)
        else:
            raise ValueError("BW not supported")


    #@tf.function
    def call(self, LLTFfield:tf.Tensor) -> tf.Tensor:
        windowSize = int((self.LLTF_length-self.LLTF_CP) / 2 * self.samplingRate)
        index = int(self.LLTF_CP * self.samplingRate)+1
        ratio = int(self.samplingRate/self.preambleBW)
        field1 = tf.signal.fftshift(tf.signal.fft(LLTFfield[:,index:index+windowSize:ratio]), axes=1)
        field2 = tf.signal.fftshift(tf.signal.fft(LLTFfield[:,index+windowSize:index+2*windowSize:ratio]), axes=1)
        
        fig,ax = plt.subplots(1, 1, figsize=(10, 6), sharex=True)
        for k in range(field1.shape[0]):
            ax.plot(tf.math.abs(field1[k,:]), label='LLTF field1')
            ax.plot(tf.math.abs(field2[k,:]), label='LLTF field2')
        ax.grid(True)
        ax.set_title("channel magnitude")
        fig.show()

        LLTF = tf.expand_dims(self.LLTF, axis=0)

        result1 = field1/LLTF
        result2 = field2/LLTF

        result = (result1 + result2)/2

        fig,ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        for k in range(field1.shape[0]):
            ax[0].plot(tf.math.abs(result1[k,:]), label='LLTF field1')
            ax[0].plot(tf.math.abs(result2[k,:]), label='LLTF field2')
        ax[0].grid(True)
        ax[0].set_title("channel estimation")

        for k in range(field1.shape[0]):
            unwrapped_phase1 = tf.math.angle(result1[k, :])
            unwrapped_phase2 = tf.math.angle(result2[k, :])
            ax[1].plot(unwrapped_phase1, label='LLTF field1')
            ax[1].plot(unwrapped_phase2, label='LLTF field2')
        ax[1].grid(True)
        fig.show()

        return result

class LSIGfield(sionna.phy.Block):
    def __init__(self, samplingRate: float):
        super().__init__()
        self.LSTF_length = 8e-6
        self.LLTF_length = 8e-6
        self.LSIG_length = 4e-6
        self.LSIG_CP = 0.8e-6
        self.samplingRate = samplingRate

    @tf.function
    def call(self, samples1:tf.Tensor, startIndex:tf.Tensor) -> tf.Tensor:
        if len(samples1.shape) == 1:
            samples1 = tf.expand_dims(samples1, axis=0)
        sample_indices = startIndex[:, tf.newaxis] + tf.cast(tf.range(int((self.LSTF_length+self.LLTF_length)* self.samplingRate), int((self.LSTF_length+self.LLTF_length+self.LSIG_length) * self.samplingRate)), startIndex.dtype)[tf.newaxis, :]

        LSIGfield = tf.gather(samples1, sample_indices, axis = 1)[0]
        return LSIGfield

class LSIGdemodulator(sionna.phy.Block):
    def __init__(self, samplingRate: float, bw: Enum):
        super().__init__()
        self.LSTF_length = 8e-6
        self.LLTF_length = 8e-6
        self.LSIG_length = 4e-6
        self.LSIG_CP = 0.8e-6
        self.samplingRate = samplingRate
        if bw == BW.BW_20MHZ:
            self.bw = 20e6
            self.pilotIndex = tf.constant([11, 25, 39, 53], dtype=tf.int32)
            self.pilotSymbol = tf.constant([1, 1, 1, -1], dtype=tf.complex64)
        elif bw == BW.BW_40MHZ:
            self.bw = 40e6

    #@tf.function
    def call(self, LSIGfield:tf.Tensor, channel:tf.Tensor) -> tf.Tensor:
        windowSize = int((self.LSIG_length-self.LSIG_CP) * self.samplingRate)
        ratio = int(self.samplingRate/self.bw)
        index = int(self.LSIG_CP * self.samplingRate)+1
        temp = tf.signal.fftshift(tf.signal.fft(LSIGfield[:,index:index+windowSize:ratio]), axes=1)

        fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        for k in range(temp.shape[0]):
            ax[0].plot(tf.math.abs(temp[k,:]))
        for k in range(temp.shape[0]):
            ax[1].plot(tf.math.angle(temp[k,:]))
        ax[0].set_title("LSIG")
        fig.show()

        temp = temp / channel
        pilot = tf.gather(temp, self.pilotIndex, axis=1)/self.pilotSymbol

        fig, ax = plt.subplots(1, 1, figsize=(10, 6), sharex=True)
        for k in range(pilot.shape[0]):
            ax.plot(tf.math.angle(pilot[k,:]),'-o')
        ax.grid(True)
        fig.show()

        pilot = tf.math.reduce_mean(pilot, axis=1)
        pilot = tf.expand_dims(pilot, axis=1)
        temp = temp / pilot
        
        fig, ax = plt.subplots(1, 1, figsize=(7, 7), sharex=True)
        for k in range(temp.shape[0]):
            ax.scatter(tf.math.real(temp[k,:]), tf.math.imag(temp[k,:]))
        ax.grid(True)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        fig.show()  
        
        return temp

class LSIGdecoder(sionna.phy.Block):
    def __init__(self):
        super().__init__()

        self.subcarrier = tf.constant([
            6,   7,  8,  9, 10, 
            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 
            22, 23, 24,     26, 27, 28, 29, 30, 31,
            33, 34, 35, 36, 37, 38,     40, 41, 42, 
            43, 44, 45, 46, 47, 48, 49, 50, 51, 52,  
                54, 55, 56, 57, 58, 
        ], dtype=tf.int32)

        self.deinterleaver = deInterleaver(1)

        self.encoder = sionna.phy.fec.conv.ConvEncoder(['1011011','1111001'])
        self.decoder = sionna.phy.fec.conv.BCJRDecoder(self.encoder)
        self.bpsk = sionna.phy.mapping.Constellation("custom", 1, [-1, 1])
        self.bpskDemapper = sionna.phy.mapping.Demapper("app","custom", 1, self.bpsk)

    #@tf.function
    def call(self, LSIGsymbols:tf.Tensor) -> tf.Tensor:

        result = self.bpskDemapper(LSIGsymbols,tf.constant(1, dtype=tf.float32))

        result = tf.gather(result, self.subcarrier, axis=1)
        result = self.deinterleaver(result)

        fig, ax = plt.subplots(1, 1, figsize=(10, 6), sharex=True)
        for k in range(result.shape[0]):
            ax.plot(result[k,:],'-o')
        ax.grid(True)
        fig.show()

        bits = self.decoder(result)

        return bits


class freqComp(sionna.phy.Block):
    def __init__(self, samplingRate: float):
        super().__init__()
        self.samplingRate = samplingRate

    @tf.function
    def call(self, samples1:tf.Tensor, freqOffset:tf.Tensor) -> tf.Tensor:
        freqOffset = tf.expand_dims(freqOffset, axis=-1)
        time_indices = tf.cast(tf.range(0, samples1.shape[1], 1, dtype=tf.float32), tf.complex64)
        cw = tf.math.exp(
            tf.cast(-2j, tf.complex64)
            * tf.cast(np.pi, tf.complex64)
            * tf.cast(freqOffset, tf.complex64) / tf.cast(self.samplingRate, tf.complex64)
            * time_indices
        )
        newSamples = samples1 * cw

        return newSamples

class deInterleaver(sionna.phy.Block):
    def __init__(self, N_CBPS: int):
        super().__init__()
        self.N_CBPS = N_CBPS

    #@tf.function
    def call(self, input1:tf.Tensor) -> tf.Tensor:
        
        batch_size = tf.shape(input1)[0]
        size = tf.shape(input1)[1]
        s = tf.maximum(self.N_CBPS // 2, 1)

        k = tf.range(size, dtype=tf.int32)
        i = tf.cast((size // 16) * (k % 16),tf.int32) + tf.cast(tf.math.floor(k / 16),tf.int32)
        j = s * tf.cast(tf.math.floor(i / s), tf.int32) + (i + size - tf.cast(tf.math.floor(16 * i / size),tf.int32)) % s

        deinterleaved = tf.gather(input1, j, axis=1)

        return deinterleaved


def unwrap(phase: tf.Tensor) -> tf.Tensor:
    phaseDiff = phase[1:] - phase[:-1]
    phaseDiff = tf.where(phaseDiff > np.pi, phaseDiff - 2 * np.pi, phaseDiff)
    phaseDiff = tf.where(phaseDiff < -np.pi, phaseDiff + 2 * np.pi, phaseDiff)
    phaseNew = tf.math.cumsum(phaseDiff, axis=0)
    return phaseNew
