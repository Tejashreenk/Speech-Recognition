from scipy.fftpack import dct
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from collections import deque
from queue import Queue
import threading
import time
import seaborn as sns
from scipy.io import wavfile
import librosa
import SpeechUtil as su
import os

def framing_and_windowing(signal, sample_rate=16000, frame_size = 0.025, frame_stride = 0.01):
    frame_size = frame_size # 25 ms
    frame_stride = frame_stride  # 10 ms
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate
    frames = np.array([signal[int(i): int(i + frame_length)] for i in range(0, len(signal) - int(frame_length) + 1, int(frame_step))])
    # By tapering the ends of each frame to zero with a smooth curve, the Hamming window reduces discontinuities at the boundaries of the frame.
    hamming = np.hamming(int(frame_length))
    windowed_frames = frames * hamming
    return windowed_frames

def fft_and_mel_filterbank(frames, sample_rate=16000, NFFT=1024):
    '''
    Compute FFT of speech window. FFT should be at least as long as the speech window (so depends on the sampling rate). Make the FFT length equal to next power of two above window length.
2 Take the magnitude of the FFT values (throwing away phase).
3 Compute Mel filter warped spectra, via overlapping triangle filters.
4 Take the log of the result.
5 Take the IDFT (or DCT) of the result.
6 Retain the first 13 coefficients, c0 through c12. Note that c0 is essentially an estimate of the log energy of the windowed speech.
    '''
    # Perform FFT
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT)) #rfft removes the negative components of fft
    mag_frames2 = np.absolute(np.fft.fft(frames, NFFT)) # absolute to drop phases. retain magnitude
    pow_frames = (((mag_frames) ** 2)/NFFT)

    # Mel Filterbanks
    fbank = mel_filterbank(NFFT)

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB
    return filter_banks

def mel_filterbank(NFFT, sample_rate=16000, num_filters=26):
    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + sample_rate / 2 / 700)
    mel_points = np.linspace(low_freq_mel, high_freq_mel, num_filters + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)
    fbank = np.zeros((num_filters, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, num_filters + 1):
        f_m_minus = int(bin[m - 1])
        f_m = int(bin[m])
        f_m_plus = int(bin[m + 1])
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    
    # fig2,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
       
    # Plot the figure
    # ax1.plot(mel_points)
    # ax1.set_title("mel_points", {'fontsize':20, 'fontweight':'bold'})
    # ax2.plot(hz_points)
    # ax2.set_title("hz_points", {'fontsize':20, 'fontweight':'bold'})
    # ax3.plot(bin)
    # ax3.set_title("bin", {'fontsize':20, 'fontweight':'bold'})
    # ax4.plot(fbank[25][:])
    # ax4.set_title("fbank", {'fontsize':20, 'fontweight':'bold'})
    # print(fbank[:][25])
    # plt.show()
    return fbank

def apply_dct(filter_banks, num_ceps = 13):
    cepstral_coeffs = dct(filter_banks, type=2, axis=1, norm='ortho')[:, :num_ceps]
    return cepstral_coeffs

def compute_mfcc(signal, sample_rate):
    windowed_frames = framing_and_windowing(signal, sample_rate)
    filter_banks = fft_and_mel_filterbank(windowed_frames, sample_rate)
    mfccs = apply_dct(filter_banks)
    return mfccs

def delta(mfccs, N=2):
    NUM_FRAMES = mfccs.shape[1]
    denominator = 2 * sum([i**2 for i in range(1, N+1)])
    delta_features = np.empty_like(mfccs)

    padded_mfccs = np.pad(mfccs, ((0, 0), (N, N)), mode='edge') # Pad mfccs along time axis
    for t in range(NUM_FRAMES):
        delta_features[:, t] = np.dot(padded_mfccs[:, t : t+2*N+1], np.arange(-N, N+1)) / denominator
    return delta_features[:, N:-N]  # compensate for padding

def calculations_for_onerecording(check,directory,filename, sample_rate = 16000):
    # su.record_save_audio(4,f"{directory}audio/{filename}")
    signal = su.read_recording(f"{directory}/{filename}",sample_rate)
    # su.display_speech_signal(signal,f"{directory}plot/{filename}")
    mfccs_librosa = librosa.feature.mfcc(y=signal, sr=sample_rate, n_fft=1024, hop_length=int(sample_rate*0.01), win_length=int(sample_rate*0.025), n_mfcc=13)

    mfcc_t = compute_mfcc(signal,sample_rate)
    mfcc = mfcc_t.transpose()
    # mfccs_librosa_t = mfccs_librosa.transpose()

    # fig3,((ax1,ax2)) = plt.subplots(1,2)
    # # Plot the figure
    # ax1.plot(mfcc[0])
    # ax1.set_xscale('log')
    # ax1.set_yscale('log')
    # ax1.set_title("mfcc", {'fontsize':20, 'fontweight':'bold'})
    # ax2.plot(mfccs_librosa_t[0])
    # ax2.set_xscale('log')
    # ax2.set_yscale('log')
    # ax2.set_title("mfccs_librosa_t", {'fontsize':20, 'fontweight':'bold'})

    # plt.show()

    # to check against librosa
    if check:
        if np.allclose(mfcc_t, mfccs_librosa.transpose()[:398]):
            print("Correct")
        else:
            print("Differences found")

    deltas = delta(mfcc, N=2)  # Compute delta features
    # print(mfcc.shape)
    features = np.concatenate((mfcc[:, 2:-2], deltas), axis=0)  # Concatenate along the feature axis
    # print(features.shape)
    return features

def calculate_mfcc_for_onerecording(check,audio_file_path, sample_rate = 16000):
    # su.record_save_audio(4,f"{directory}audio/{filename}")
    audio_file_path = str(audio_file_path).replace('.wav','')
    signal = su.read_recording(audio_file_path,sample_rate)
    # su.display_speech_signal(signal,f"{directory}plot/{filename}")
    mfccs_librosa = librosa.feature.mfcc(y=signal, sr=sample_rate, n_fft=1024, hop_length=int(sample_rate*0.01), win_length=int(sample_rate*0.025), n_mfcc=13)

    mfcc_t = compute_mfcc(signal,sample_rate)
    mfcc = mfcc_t.transpose()
    # mfccs_librosa_t = mfccs_librosa.transpose()

    # fig3,((ax1,ax2)) = plt.subplots(1,2)
    # # Plot the figure
    # ax1.plot(mfcc[0])
    # ax1.set_xscale('log')
    # ax1.set_yscale('log')
    # ax1.set_title("mfcc", {'fontsize':20, 'fontweight':'bold'})
    # ax2.plot(mfccs_librosa_t[0])
    # ax2.set_xscale('log')
    # ax2.set_yscale('log')
    # ax2.set_title("mfccs_librosa_t", {'fontsize':20, 'fontweight':'bold'})

    # plt.show()

    # to check against librosa
    if check:
        if np.allclose(mfcc_t, mfccs_librosa.transpose()[:398]):
            print("Correct")
        else:
            print("Differences found")

    deltas = delta(mfcc, N=2)  # Compute delta features
    # print(mfcc.shape)
    features = np.concatenate((mfcc[:, 2:-2], deltas), axis=0)  # Concatenate along the feature axis
    # print(features.shape)
    return features


def plot_features(features):
    # Plot MFCC and Delta features matrix
    plt.figure(figsize=(12, 8))
    librosa.display.specshow(features, sr=sample_rate, x_axis='time', hop_length=int(sample_rate*0.01))
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Librosa MFCC and Delta Features for {filename}')
    plt.xlabel('Time (frames)')
    plt.ylabel('MFCC + Delta Coefficients')
    plt.savefig(f'{directory}librossa/{filename}_features.png')
    # plt.show()

    # Use imshow to replicate MATLAB's imagesc functionality
    fig, ax = plt.subplots(figsize=(12, 8))

    # Display the data as a heatmap with origin at bottom left
    cax = ax.imshow(features, cmap='viridis', aspect='auto', origin='lower')

    # Add a color bar to the figure, associated with the displayed image
    fig.colorbar(cax, ax=ax)

    # Add title and labels
    ax.set_title(f'MFCC and Delta Features for {filename}')
    ax.set_xlabel('Time Frame Index')
    ax.set_ylabel('MFCC + Delta Coefficients')

    # Save the plot to a file
    plt.savefig(f'{directory}calc_mfcc/{filename}_features_imsc.png')

    # Show the plot
    # plt.show()


# directory = 'cleaned_audios_simple'

# # List all files in the directory
# files_in_directory = os.listdir(directory)
# sample_rate = 16000
# features_arr = []

# for filename in files_in_directory:
#     features = calculations_for_onerecording(False, directory=directory,filename=filename,sample_rate=sample_rate)
#     features_arr.append(features)

