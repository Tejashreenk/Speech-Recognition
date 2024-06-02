import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from collections import deque
import threading

class AudioMonitor_LamelRabiner:
    
    def __init__(self, device=0, fs=16000, block_duration_ms=10):
        self.device = device
        self.fs = fs
        self.block_size = int(fs * block_duration_ms / 1000)
        self.energy = deque(maxlen=400)  # Stores last 400 blocks
        self.zcr = deque(maxlen=400)
        self.lock = threading.Lock()
        self.audio = deque(maxlen=400)
        self.threshold_values = []
        self.log_energy = []

        self.Q = 0

    def process_saved_audio(self,signal):
        # self.threshold_values = self.detect_speech_start(signal)
        # Create chunks
        chunks = [signal[i:i+self.block_size] for i in range(0,len(signal),self.block_size)]
        for audio in chunks:
            self.log_energy.append(self.compute_short_time_log_energy(audio))
        self.Q = self.estimate_noise_floor()
        # Adjusted log energy
        adjusted_log_energy = self.log_energy - self.Q
        
        return adjusted_log_energy


    def compute_short_time_log_energy(self,signal):
        """ Compute short-time energy of a signal """
        # Energy as the sum of abs of the signal
        energy = np.sum(np.abs(signal))
        # Compute log energy, adding small constant to avoid log(0)
        log_energy = np.log(energy + 1e-8)

        return log_energy

    def compute_short_time_log_energy_square(self,signal):
        energy = np.sum(np.abs(signal) ** 2)
        # Compute log energy, adding small constant to avoid log(0)
        log_energy = np.log(energy + 1e-8)

        return log_energy
    
    def estimate_noise_floor(self,bin_num=50):
        # print(self.log_energy)

        """Estimate the noise floor from log energy using a histogram within the lowest 10 dB."""
        Emin = np.min(self.log_energy)
        # print(Emin)
        Emax = Emin + 10  # Emin + 10 dB in logarithmic scale
        hist, bin_arr = np.histogram(self.log_energy, bins=bin_num, range=(Emin, Emax)) #default number of bins = 10 which is not good
        
        '''
        The choice of 50 allows for a detailed view of the distribution in 10 dB 
        10/50 = granularity of 0.2 dB per bin, which can be very informative for audio energy levels.
        Three-point averaging is a sliding window average with a window size = 3 
        It is a simple smoothing technique used in signal processing and data analysis. 
        This method averages each point in a dataset with its immediate neighbors 
        to smooth out short-term fluctuations and highlight longer-term trends or patterns.
        '''
        # Smooth histogram with a three-point average
        smooth_hist = np.convolve(hist, np.ones(3)/3, mode='same')
        
        # Find the peak of the histogram
        peak_idx = np.argmax(smooth_hist)
        '''
        The level equalized energy array has the property that during silence it fluctuates around the0 dB level, 
        and during speech it is considerably large.
        '''
        Q = bin_arr[peak_idx]  # Noise floor estimate
        
        return Q
    
    def plot_data(self, adjusted_log_energy,filename):
        plt.figure(figsize=(10, 4))
        plt.plot(adjusted_log_energy, label='Adjusted Log Energy')
        # plt.axhline(0, color='red', linestyle='--', label='Silence Threshold')
        plt.legend()
        plt.title('Adjusted Log Energy of Audio Signal')
        plt.xlabel('Frame Index')
        plt.ylabel('Adjusted Log Energy')
        plt.savefig(f'{filename}.png')
        # plt.show()
