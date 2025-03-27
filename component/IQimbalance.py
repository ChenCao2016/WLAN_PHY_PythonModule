import cmath
from .DFT import *
from .GlobeParameter import *
from scipy.fft import fft, ifft, fftshift, fftfreq
import numpy as np

from matplotlib import pyplot as plt


# ----------------------------------------------------------------------
# Estimate IQ imbalance on L-STF
# amplitude imbalance and phase imbalance only
# ----------------------------------------------------------------------
# INPUT
# samples: complex number array
# samplingRate: number (Hz)
# LSTF_endIndex: sample index array for LSTF end index
# ----------------------------------------------------------------------
# OUTPUT
# IQimbalance: IQ imbalance estimated on L-STF
# ----------------------------------------------------------------------
# Tubbax, Jan, et al. "Compensation of IQ imbalance and phase noise in OFDM systems." IEEE Transactions on Wireless Communications 4.3 (2005): 872-877.

def IQimbalance_LLTF(samples, samplingRate, LSTF_endIndex, LLTF_channel):


    amplitudeImbalance = 0.0
    phaseImbalance = 0.0

    alpha = np.cos(phaseImbalance) + 1j*amplitudeImbalance*np.sin(phaseImbalance)
    beta = amplitudeImbalance*np.cos(phaseImbalance) - 1j*np.sin(phaseImbalance)

    channleMirror = np.conj(LLTF_channel)
    channleMirror = channleMirror[::-1]
    channleMirror = np.roll(channleMirror, 1)

    LLTFmirror = np.array(LLTF)
    LLTFmirror = LLTFmirror[::-1]
    LLTFmirror = np.roll(LLTFmirror, 1)

    corrected = alpha*np.array(LLTF_channel) - beta*np.array(LLTF)*LLTFmirror*channleMirror
    corrected = corrected/(np.abs(alpha)**2 - np.abs(beta)**2)
    
    fig, ax = plt.subplots(2,2)
    ax[0,0].plot(np.abs(LLTF))
    ax[0,0].plot(np.abs(LLTFmirror))

    ax[0,1].plot(np.angle(LLTF))
    ax[0,1].plot(np.angle(LLTFmirror))

    ax[1,0].plot(np.abs(LLTF_channel))
    ax[1,0].plot(np.abs(channleMirror))
    ax[1,0].plot(np.abs(corrected))

    ax[1,1].plot(np.angle(LLTF_channel))
    ax[1,1].plot(np.angle(channleMirror))
    ax[1,1].plot(np.angle(corrected))
    fig.show()


    correctedShifted = np.roll(corrected, 1)
    
    fig, ax = plt.subplots(2,1)
    ax[0].plot(np.abs(corrected))
    ax[0].plot(np.abs(correctedShifted))
    ax[1].plot(np.angle(corrected))
    ax[1].plot(np.angle(correctedShifted))
    fig.show()

    mse = np.abs(corrected - correctedShifted)**2
    fig, ax = plt.subplots(2,1)
    ax[0].plot(mse)
    ax[1].plot(mse[7:32])
    ax[1].plot(mse[34:59])
    fig.show()


    result = []
    searchAmplitude = np.linspace(-0.02, 0.02, 200)
    searchPhase = np.linspace(-0.02, 0.02, 200)
    
    for phaseImbalance in searchAmplitude:
        for amplitudeImbalance in searchPhase:
            alpha = np.cos(phaseImbalance) + 1j*amplitudeImbalance*np.sin(phaseImbalance)
            beta = amplitudeImbalance*np.cos(phaseImbalance) - 1j*np.sin(phaseImbalance)
            corrected = alpha*np.array(LLTF_channel) - beta*np.array(LLTF)*LLTFmirror*channleMirror
            corrected = corrected/(np.abs(alpha)**2 - np.abs(beta)**2)
            correctedShifted = np.roll(corrected, 1)
            mse = np.abs(corrected - correctedShifted)**2
            result.append(np.mean(mse[7:32]) + np.mean(mse[34:59]))

    # Reshape the result into a 2D array
    result_2d = np.array(result).reshape(len(searchPhase), len(searchAmplitude))

    X, Y = np.meshgrid(searchAmplitude, searchPhase)

    min_index = np.unravel_index(np.argmin(result_2d), result_2d.shape)
    min_amplitude = searchAmplitude[min_index[1]]
    min_phase = searchPhase[min_index[0]]
    min_value = result_2d[min_index]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, result_2d, cmap='viridis')

    ax.scatter(min_amplitude, min_phase, min_value, color='red', s=50, label='Min Point')
    ax.legend()

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    ax.set_xlabel("Amplitude Imbalance")
    ax.set_ylabel("Phase Imbalance")
    ax.set_zlabel("MSE")
    ax.set_title("MSE Surface Plot")

    ax.text(min_amplitude, min_phase, min_value, 
            f"Min: ({min_amplitude:.4f}, {min_phase:.4f}, {min_value:.4f})", 
            color='red')

    fig.show()


    alpha = np.cos(min_phase) + 1j*min_amplitude*np.sin(min_phase)
    beta = min_amplitude*np.cos(min_phase) - 1j*np.sin(min_phase)
    corrected = alpha*np.array(LLTF_channel) - beta*np.array(LLTF)*LLTFmirror*channleMirror
    corrected = corrected/(np.abs(alpha)**2 - np.abs(beta)**2)


    fig, ax = plt.subplots(2,1)
    ax[0].plot(np.abs(LLTF_channel[6:31]))
    ax[0].plot(np.abs(corrected[6:31]))

    ax[1].plot(np.angle(LLTF_channel[6:31]))
    ax[1].plot(np.unwrap(np.angle(corrected[6:31])))
    fig.show()


    min_amplitude = 1-min_amplitude
    IMRR = 10*np.log10((1+min_amplitude**2+2*min_amplitude*np.cos(min_phase))/(1+min_amplitude**2-2*min_amplitude*np.cos(min_phase)))
    
    print(f"Estimated Amplitude Imbalance: {20*np.log10(min_amplitude)} dB")
    print(f"Estimated Phase Imbalance: {min_phase/np.pi*360} degree")
    print(f"Estimated IMRR: {IMRR} dB")




    return 0