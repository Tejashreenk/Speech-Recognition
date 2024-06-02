import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.io import wavfile

def record_save_audio(time, filename, Fs = 16000):
    # Record audio for 4 seconds
    print(f"Start speaking for {filename}! Recording for {time} seconds...")
    s = sd.rec(int(time * Fs), samplerate=Fs, channels=1, dtype='float64')
    sd.wait()  # Wait until the recording is finished
    print("Recording finished.")

    # Save the original recording
    wavfile.write(f'{filename}.wav', Fs, s)

def save_signal(signal, filename, Fs = 16000):
    # Save the original recording
    wavfile.write(f'{filename}.wav', Fs, signal)

def play_recording(filename, Fs=16000):
    # Read and play back the original recording
    print("Playing re-read in PCM recording...")
    s_read = wavfile.read(f'{filename}.wav')[1]
    sd.play(s_read, Fs)
    sd.wait()
    return s_read 

def play_signal(signal, Fs=16000):
    # Read and play back the original recording
    print("Playing signal...")
    sd.play(signal, Fs)
    sd.wait()

def read_recording(filename, Fs=16000):
    # Read and play back the original recording
    # print(f"Read in PCM recording...{filename}")
    s_read = wavfile.read(f'{filename}.wav')[1]
    # sd.play(s_read, Fs)
    sd.wait()
    return s_read 

def display_speech_signal(s_read,filename,Fs=16000):
    # Plot the time-domain waveform of the sound
    plt.figure(figsize=(10, 4))
    plt.plot(np.linspace(0, len(s_read) / Fs, num=len(s_read)), s_read)
    plt.title('Time-Domain Waveform')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.savefig(f'{filename}.png')
    # plt.show()
    plt.close()
