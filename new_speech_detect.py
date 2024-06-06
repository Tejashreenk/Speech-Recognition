import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from collections import deque
from queue import Queue, Empty
import threading
import time
import matplotlib
from scipy.io import wavfile
import librosa
import SpeechUtil as su
import os
import pickle
import HMM_final
import HMM_lib
# matplotlib.use('png')

# Queue to hold data for plotting
data_queue = Queue()
# Queue to compute data for plotting
compute_queue = Queue()
i=0



class AudioMonitor:
    def __init__(self, device=0, fs=16000, block_duration_ms=10):
        self.device = device
        self.fs = fs
        self.block_size = int(fs * block_duration_ms / 1000)
        self.energies = []
        self.zcrs = []
        self.energy = deque(maxlen=400)  # Stores last 400 blocks
        self.zcr = deque(maxlen=400)
        self.lock = threading.Lock()
        self.audio = deque(maxlen=400)
        self.threshold_values = []

    def compute_short_time_energy(self,signal):
        """ Compute short-time energy of a signal """
        # Energy as the sum of abs of the signal
        return np.sum(np.abs(signal))

    def compute_short_time_energy_square(self,signal):
        return np.sum(np.abs(signal) ** 2)

    def compute_zero_crossing_rate_sum(self,signal):
        """ Compute zero-crossing rate of a signal using the given formula """

        # Calculate sign changes
        signs = np.sign(signal)
        signs[signs == 0] = 1  # Treat zero as positive sign
        sign_changes = np.abs(np.diff(signs))/2

        zcr = np.sum(sign_changes)
        return zcr

    def compute_zero_crossing_rate(self,signal):
        """ Compute zero-crossing rate of a signal using the given formula """
        # Calculate sign changes
        signs = np.sign(signal)
        signs[signs == 0] = 1  # Treat zero as positive sign
        sign_changes = np.abs(np.diff(signs))/2

        # zcr = np.sum(sign_changes)
        return sign_changes

    def start(self):
        with sd.InputStream(device=self.device, channels=1, callback=self.audio_callback,
                            blocksize=self.block_size, samplerate=self.fs, dtype=np.float32):
            print("Starting streaming...")
            threading.Thread(target=self.plot_data).start()
            sd.sleep(1000)  # Stream for a long time, you can adjust as needed

    def detect_if_noise(self, energy):
        # Step 1: Compute Es(n)
        energy = energy
        # Step 2: Compute IMX, IMN
        IMX = np.max(energy)
        if IMX<4.5:
            # The audio is noise
            return True
        else:
            return False

    def detect_speech_start(self, signal, frame_ms=10):
        frame_length = self.block_size

        # Step 1: Compute Es(n)
        energy = self.energies
        zcr = self.zcrs
        # Step 2: Compute IMX, IMN
        IMX, IMN = np.max(energy), np.min(energy)

        if IMX<3:
            # The audio is noise
            return [0, 0,0, 0, IMX, 0, 0 ]
        # Step 3: Compute thresholds ITL, ITU
        ITL = min(0.03 * (IMX - IMN) + IMN, 4 * IMN)
        ITU = 5 * ITL

        # Step 4: Define IZCT
        initial_zcr =  []
        first_100ms = int(0.1 * self.fs)  # Number of samples in first 100ms
        first_100ms_chunks = [signal[i:i+self.block_size] for i in range(0,first_100ms,self.block_size)]
        for audio in first_100ms_chunks:
            initial_zcr.append(self.compute_zero_crossing_rate_sum(audio))
        IZC, sigma_IZC = np.mean(initial_zcr), np.std(initial_zcr)
        IF = 25
        IZCT = min(IF, IZC + 2 * sigma_IZC)

        # Step 5: Find N1
        above_ITL = np.where(energy > ITL)[0]
        # print(f" IMX:{IMX}, IMN:{IMN},ITL:{ITL},ITU:{ITU},IZCT:{IZCT}  ")

        # Iterate through entire energies
        # for e in energy:
        N1 = 0
        # print(f"above_ITL:{above_ITL} ")
        for i in above_ITL:
            if all(energy[j] > ITL for j in range(i, min(i + 25, len(energy)))) and all(energy[j] > ITU for j in range(i, min(i + 7, len(energy)))):
                N1 = i
                break
        
        # Step 5: Find N1 Logic2
        above_ITL = np.where(energy > ITL)[0]
        above_ITU = np.where(energy > ITU)[0]
        # print(f"above_ITU:{above_ITU} ")
        above_ITL_diff = np.diff(above_ITL)
        above_ITU_diff = np.diff(above_ITU)
        # print(f"above_ITU_diff:{above_ITU_diff} len: {len(above_ITU_diff)}")
        # print(f"above_ITU:{above_ITU} len: {len(above_ITU)}")
        # print(f"above_ITL_diff:{above_ITL_diff}")
        # print(f"above_ITL:{above_ITL}")
        below_ITU = []
        for i in range(1,len(above_ITU)):
            j = 1
            if above_ITU[i]-above_ITU[i-1] > min(j,10):
                if np.all(energy[above_ITU[i]:above_ITU[i]+5] < ITL):
                    # energy falls below ITL for 5s
                    # N2 = 
                    below_ITU.append(i)
            elif above_ITU[i]-above_ITU[i-1] == 1:
                j+=1
        
        # print(f"above_ITU:{below_ITU} ")

        # i,j = 0
        # while above_ITL[i]<above_ITU[j] and above_ITU[j+7]==above_ITU[j]+7 and j<len(above_ITU)-7 and i<len(above_ITL):
        #     N1 = above_ITL[i]
        #     i+=1
        # print(f"N1:{N1} ")
        # Step 6: Adjust N1 if needed to N1'
        search_range = range(max(0, N1 - 25), N1 + 1)
        zcr = zcr
        high_zcr_frames = [i for i in search_range if zcr[i] > IZCT]
        if len(high_zcr_frames) >= 3:
            N1_prime = high_zcr_frames[0]
        else:
            N1_prime = N1
        
        # Step 7: Find N2 Logic
        # N2 = N1
        # for i in above_ITU:
        #     if any(energy[j] < ITL for j in range(i, min(i + 25, len(energy)))) and all(energy[j] < ITU for j in range(i, min(i + 15, len(energy)))):
        #         N2 = i
        #         break
        j = above_ITU[-1]
        l=-1
        while ((above_ITU[l] - above_ITU[l-1]>1) or (above_ITU[l-1] - above_ITU[l-2] > 1)) and abs(l-2)<len(above_ITU):
            l -= 1
            j = above_ITU[l]

        index = np.array(above_ITL).searchsorted(j)
        value = above_ITL[index]
        while len(above_ITL)>value and above_ITL[index] == value:
            value += 1
            index += 1
        N2 = value

        # Step 8: Adjust N2 if needed to N2'
        search_range = range( N2 + 1,min(N2 + 25,len(energy)))
        high_zcr_frames = [i for i in search_range if zcr[i] > IZCT]
        if len(high_zcr_frames) >= 3:
            N2_prime = high_zcr_frames[-1]
        else:
            N2_prime = N2

        return_values = [N1_prime, N1,N2, N2_prime, ITU, ITL, IZCT ]
        return return_values  # Returning values for plotting

    def detect_speech_start_2(self, signal, frame_ms=10):
        frame_length = self.block_size

        # Step 1: Compute Es(n)
        energy = self.energies
        zcr = self.zcrs
        # Step 2: Compute IMX, IMN
        IMX, IMN = np.max(energy), np.min(energy)

        if IMX<4.5:
            # The audio is noise
            return [[[0,0,0,0]], IMX, 0, 0 ]
        # Step 3: Compute thresholds ITL, ITU
        ITL = min(0.03 * (IMX - IMN) + IMN, 4 * IMN)
        ITU = 5 * ITL

        # Step 4: Define IZCT
        initial_zcr =  []
        first_100ms = int(0.1 * self.fs)  # Number of samples in first 100ms
        first_100ms_chunks = [signal[i:i+self.block_size] for i in range(0,first_100ms,self.block_size)]
        for audio in first_100ms_chunks:
            initial_zcr.append(self.compute_zero_crossing_rate_sum(audio))
        IZC, sigma_IZC = np.mean(initial_zcr), np.std(initial_zcr)
        IF = 25
        IZCT = min(IF, IZC + 2 * sigma_IZC)

        # Step 5: Find N1
        above_ITL = np.where(energy > ITL)[0]
        # print(f" IMX:{IMX}, IMN:{IMN},ITL:{ITL},ITU:{ITU},IZCT:{IZCT}  ")
       
        # Iterate through entire energies
        # for e in energy:
        N2_crossing = []
        N1_crossing = []
        c=0
        d=0
        N2_start = -1
        N1_start = -1
        # print(f"above_ITL:{above_ITL} ")
        for i,e in enumerate(energy): 
            if e>ITU:
                if N2_start == -1:
                    N2_start = i
                c+=1
                if i == len(energy)-1:
                    N2_crossing.append([N2_start,c])
                    N2_start = -1
                    c=0
            else:
                if c>5:
                    N2_crossing.append([N2_start,c])
                    N2_start = -1
                    c=0
                elif N2_start == 0:
                    N2_crossing.append([N2_start,c])
                    N2_start = -1
                    c=0
                else:
                    N2_start = -1
                    c=0
            
            if e > ITL:
                if N1_start == -1:
                    N1_start = i
                d+=1
                if i == len(energy)-1 and len(N2_crossing)>len(N1_crossing):
                    if N1_start <= N2_crossing[-1][0]:
                        N1_crossing.append([N1_start,d])
                        N1_start = -1
                        d=0
                    else:
                        print(N1_start,d)
            else:
                if d>5 and len(N2_crossing)>len(N1_crossing) and N1_start <= N2_crossing[-1][0]:
                    N1_crossing.append([N1_start,d])
                    N1_start = -1
                    d=0
                else:
                    N1_start = -1
                    d=0
             

        # print(f"N2_crossing:{N2_crossing} len: {len(N2_crossing)}")
        # print(f"N1_crossing:{N1_crossing} len: {len(N1_crossing)}")
        N_array = []
        if len(N1_crossing)>0:
            # Calc for N1_prime N2_prime
            for value in N1_crossing:
                N1 = value[0]
                N2 = N1 + value[1]
                N1_prime = max(self.calc_N1_prime(N1,IZCT,zcr)-10,0) # -10ms from the start to get silence before spoken word
                N2_prime = self.calc_N2_prime(N2,IZCT,len(energy),zcr)
                N_array.append([N1_prime,N1,N2,N2_prime])
        else: 
            # The audio is noise
            return [[[0,0,0,0]], IMX, 0, 0 ]
            
        return_values = [N_array, ITU, ITL, IZCT]
        return return_values  # Returning values for plotting

    def calc_N1_prime(self, N1, IZCT, zcr):
        # Step 6: Adjust N1 if needed to N1'
        search_range = range(max(0, N1 - 25), N1 + 1)
        zcr = zcr
        high_zcr_frames = [i for i in search_range if zcr[i] > IZCT]
        if len(high_zcr_frames) >= 3:
            N1_prime = high_zcr_frames[0]
        else:
            N1_prime = N1
        
        return N1_prime
    
    def calc_N2_prime(self, N2,IZCT,len_energy,zcr):
        # Step 8: Adjust N2 if needed to N2'
        search_range = range( N2 + 1,min(N2 + 25,len_energy))
        high_zcr_frames = [i for i in search_range if zcr[i] > IZCT]
        if len(high_zcr_frames) >= 3:
            N2_prime = high_zcr_frames[-1]
        else:
            N2_prime = N2 

        return N2_prime
        
    def start_saved_audio(self,signal):
        # self.threshold_values = self.detect_speech_start(signal)
        # Create chunks
        chunks = [signal[i:i+self.block_size] for i in range(0,len(signal),self.block_size)]

        for audio in chunks:
            energy, zcr = self.audio_callback(audio,0,0,0)
            self.energies.append(energy)
            self.zcrs.append(zcr)

        is_noise = self.detect_if_noise(self.energies)
        if is_noise != True:
            self.threshold_values  = self.detect_speech_start_2(signal,self.fs)
        return self.threshold_values
        


    def audio_callback(self, indata, frames, time, status):
        # print("processing audio ...")
        if status:
            print("Stream status:", status)
        # signal = indata[:, 0]

        signal = indata[:]
        energy = self.compute_short_time_energy(signal)
        zcr = self.compute_zero_crossing_rate_sum(signal)
        # print(zcr)
        return energy, zcr

    def compute_spectrogram(self,audio, sr):
        # Generate a Short-Time Fourier Transform (STFT) spectrogram
        S = np.abs(librosa.stft(audio))
        log_S = librosa.amplitude_to_db(S, ref=np.max)
        return log_S, sr

    def plot_spectrogram(self,filename, audio, sr=16000):
        # signal = audio
        # chunks = [signal[i:i+self.block_size] for i in range(0,len(signal),self.block_size)]
        # for audio in chunks:
        #     self.audio_callback(audio,0,0,0)

        plt.figure(figsize=(12, 4))
        # Load audio file
        y, sr = librosa.load(f'{directory}/recordings/{filename}.wav',sr = sr)

        # Generate spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', cmap='coolwarm')
        plt.title('Spectrogram')
        plt.colorbar(format='%+2.0f dB')

        # # Annotate speech segments on the spectrogram
        # for start, end in speech_segments:
        #     # ax.axvline(x=start, color='r', linestyle='--')  # Start of speech
        #     # ax.axvline(x=end, color='b', linestyle='--')   # End of speech
        # print(self.threshold_values)
        plt.axvline(x=self.threshold_values[0]/100, color='b', linestyle='--', linewidth=2, label='N1_prime')
        plt.axvline(x=self.threshold_values[1]/100, color='g', linestyle='--', linewidth=2, label='N1')
        plt.axvline(x=self.threshold_values[2]/100, color='g', linestyle='--', linewidth=2, label='N2')
        plt.axvline(x=self.threshold_values[3]/100, color='b', linestyle='--', linewidth=2, label='N2_prime')
        plt.legend()
        plt.pause(0.1)

        plt.savefig(f'spectrogram_{filename}.png')
        plt.close()

        # return img


    def plot_data(self,filename):
        plt.ion()
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        # print(self.zcrs)

        if self.energies and self.zcrs and self.threshold_values:
            axs[0].cla()
            axs[0].plot(self.energies)
            axs[0].set_title("Energy")
            axs[0].axhline(y=self.threshold_values[-2], color='r', linestyle='--', linewidth=2, label='ITL')
            axs[0].axhline(y=self.threshold_values[-3], color='g', linestyle='--', linewidth=2, label='ITU')
            axs[0].axvline(x=self.threshold_values[0], color='b', linestyle='-', linewidth=2, label='N1_prime')
            axs[0].axvline(x=self.threshold_values[1], color='g', linestyle='-', linewidth=2, label='N1')
            axs[0].axvline(x=self.threshold_values[2], color='g', linestyle='--', linewidth=2, label='N2')
            axs[0].axvline(x=self.threshold_values[3], color='b', linestyle='--', linewidth=2, label='N2_prime')
            axs[0].legend()

            axs[1].cla()
            axs[1].plot(self.zcrs)
            axs[1].set_title("Zero-Crossing Rate")
            axs[1].axhline(y=self.threshold_values[-1], color='r', linestyle='--', linewidth=2, label='IZCT')
            axs[1].legend()
            plt.pause(0.1)
        # plt.show()
        plt.savefig(f'{filename}.png')
        # time.sleep(0.1)  # Adjust the sleep time to control plot update rate
        plt.close()

    def plot_data_new(self,filename):
        plt.ion()
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        # print(self.zcrs)

        if self.energies and self.zcrs and self.threshold_values:
            axs[0].cla()
            axs[0].plot(self.energies)
            axs[0].set_title("Energy")
            axs[0].axhline(y=self.threshold_values[-2], color='r', linestyle='--', linewidth=2, label='ITL')
            axs[0].axhline(y=self.threshold_values[-3], color='g', linestyle='--', linewidth=2, label='ITU')
            for threshold in self.threshold_values[0]:
                print(threshold)
                axs[0].axvline(x=threshold[0], color='b', linestyle='-', linewidth=2, label='N1_prime')
                axs[0].axvline(x=threshold[1], color='g', linestyle='-', linewidth=2, label='N1')
                axs[0].axvline(x=threshold[2], color='g', linestyle='--', linewidth=2, label='N2')
                axs[0].axvline(x=threshold[3], color='b', linestyle='--', linewidth=2, label='N2_prime')
            axs[0].legend()

            axs[1].cla()
            axs[1].plot(self.zcrs)
            axs[1].set_title("Zero-Crossing Rate")
            axs[1].axhline(y=self.threshold_values[-1], color='r', linestyle='--', linewidth=2, label='IZCT')
            axs[1].legend()
            plt.pause(0.1)
        # plt.show()
        plt.savefig(f'{filename}.png')
        # time.sleep(0.1)  # Adjust the sleep time to control plot update rate
        plt.close()

    def combine_audio(self, signal, thresholds, sample_rate=16000):
        start =  0
        end = 0
        i = 0
        signal_arr = []
        while i<len(thresholds):
            if min(thresholds[i]) == max(thresholds[i]) == 0:
                return []
            if end != thresholds[i][3]:
                start = thresholds[i][0]
                end = thresholds[i][3]
            if i+1 < len(thresholds) and thresholds[i+1][0]<end:
                end = thresholds[i+1][3]
                i+=1
            else:
                # print(start,end)
                signal_arr.append(signal[int(start * sample_rate*0.01):int(end * sample_rate*0.01)]) #10ms frames
                i+=1 
                    
        sg_conc = np.concatenate(signal_arr)
        # su.play_signal(sg_conc)
        # time.sleep(1)
    
        return sg_conc

    def combine_audio_simple(self, signal, thresholds, sample_rate=16000):
        
        if len(thresholds)>0:
            if min(thresholds[0]) == max(thresholds[0]) == 0:
                return []
            else:
                return signal
        
# Background thread to record audio continuously
# Chunk the audio in 2 ms chunks
# process the chunks on mainthread

# Constants
CHANNELS = 1  # Mono audio
CHUNK_SIZE_TIME = 1
SAMPLE_RATE = 22050#16000##44100 # Sample rate in Hz
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_SIZE_TIME)  # 2 s chunk size at 44100 Hz
audio_sg = []
silence_count = 0

# Create a thread-safe queue
audio_queue = Queue()
chunked_audio_queue = Queue()

def audio_callback(indata, frames, time, status):
    """This function is called for each audio block."""
    if status:
        print(status)
    # Put the incoming audio chunk into the queue
    audio_queue.put(indata[:, 0].copy())

def record_audio():
    """Continuously record audio in chunks and send to the main thread."""
    with sd.InputStream(callback=audio_callback, dtype='float32', 
                        channels=CHANNELS, samplerate=SAMPLE_RATE, blocksize=CHUNK_SIZE):
        input("Press Enter to stop recording...\n")

def process_recorded_audio(signal,i,filename,directory):
    chunks = [signal[i:i+CHUNK_SIZE] for i in range(0,len(signal),CHUNK_SIZE)]
    i=0
    for chunk in chunks:
        # audio_queue.put(chunk[:].copy())
        i+=1
        process_chunk(chunk,i,directory,is_livestream)
    get_cleaned_audio(i,filename,directory)


def process_audio(i,is_livestream,directory):
    """Process audio chunks from the queue."""
    while audio_queue.not_empty:
        try:
            # Get the next audio chunk from the queue
            chunk = audio_queue.get(True, timeout=1)  # Adjust timeout as needed
            su.save_signal(chunk,f"recorded_audios/audio_{i}",SAMPLE_RATE)
            process_chunk(chunk,i,directory,is_livestream)
            i+=1


        except Empty:
            # if audio_sg:
            #     su.save_signal(np.array(audio_sg).flatten(),f"cleaned_audios/audio_{j}",SAMPLE_RATE)
            #     audio_sg=[]
            # get_cleaned_audio(i,"")
            print("No audio data")

# def process_recorded_audio(i,is_livestream):
#     """Process audio chunks from the queue."""
#     while audio_queue.not_empty:
#         try:
#             # Get the next audio chunk from the queue
#             chunk = audio_queue.get(True, timeout=2)  # Adjust timeout as needed
#             su.save_signal(chunk,f"recorded_audios/audio_{i}",SAMPLE_RATE)
#             process_chunk(chunk,i,is_livestream)
#             i+=1


#         except Empty:
#             # if audio_sg:
#             #     su.save_signal(np.array(audio_sg).flatten(),f"cleaned_audios/audio_{j}",SAMPLE_RATE)
#             #     audio_sg=[]
#             get_cleaned_audio(i)
#             print("No audio data received for processing.")

def process_chunk(chunk,i,directory,is_livestream = True):
    global silence_count  # Declare that we are using the global variable

    """Process an individual chunk of audio data."""
    monitor = AudioMonitor(device=0, fs=SAMPLE_RATE)
    thresholds = monitor.start_saved_audio(chunk)
    print(f"thresholds {i} : {thresholds}")
    # monitor.plot_data_new(f"plot/audio_{i}")
    # monitor.plot_spectrogram(f"{audio}",signal)
    filename = f"audio_{i}"
    if len(thresholds)>0 and len(thresholds[0])>0 and thresholds[0][0][-1]>0:
        # audio_sg=(monitor.combine_audio_simple(chunk, thresholds[0],SAMPLE_RATE))
        audio_sg=(monitor.combine_audio(chunk, thresholds[0],SAMPLE_RATE))
        # su.save_signal(audio_sg,f"cleaned_audios/audio_{j}",SAMPLE_RATE)
        if len(audio_sg) > 0:
            chunked_audio_queue.put(audio_sg[:].copy())
    else:
        # silence detected
        silence_count += 1
    # print(silence_count)
    if is_livestream and silence_count>0:
        get_cleaned_audio(i,filename,directory)
        silence_count=0
    # else:
    #     get_cleaned_audio(i)



def get_cleaned_audio(j,filename,directory):
    """Process audio chunks from the queue."""
    chunks = []  # Initialize chunks list outside the loop

    while not chunked_audio_queue.empty():  # Proper method to check if queue is empty
        try:
            # Get the next audio chunk from the queue
            chunk = chunked_audio_queue.get(True, timeout=1)  # Adjust timeout as needed
            chunks.append(chunk)
            # print(len(chunks))
        except Empty:
            print("Timeout reached; no more audio data.")

    # Flatten the collected chunks and save the resulting audio
    if chunks:  # Check if there are any collected chunks to process
        combined_audio = np.concatenate(chunks)  # Use concatenate for better performance with NumPy arrays
        su.save_signal(combined_audio, f"{directory}/{filename}", SAMPLE_RATE)
        # print(f"Combined audio saved as {directory}/{filename}.wav")
        recognized_label = HMM_lib.recognize_lib(loaded_models, f"{directory}/{filename}.wav",SAMPLE_RATE)
        print(f'Recognized as: {recognized_label}')
    # else:
        # print("No audio chunks were processed.")

def load_models(model_path,model_dir='models'):
    models = {}
    for file in os.listdir(model_dir):
        if file.endswith(f'{model_path}'):
            label = file.replace(f'{model_path}', '')
            curr_model_path = os.path.join(model_dir, file)
            with open(curr_model_path, 'rb') as f:
                models[label] = pickle.load(f)
            print(f'Model for {label} loaded from {curr_model_path}')
    return models

if __name__ == "__main__":
    is_livestream = True
    model_path = "_hmm_model_lib.pkl"
    # Load models
    loaded_models = load_models(model_path,"SavedModels")
    clean_audio_directory = "cleaned_audios_3005"
    # Recognize a new audio file

    if is_livestream:
        # Start the recording in a background thread
        threading.Thread(target=record_audio, daemon=True).start()
        time.sleep(CHUNK_SIZE_TIME)
        i=0

        # Process the audio on the main thread
        process_audio(i,is_livestream,clean_audio_directory)

    else:
        # Specify the directory you want to list
        directory_path = 'Recordings_3005'

        # List all files in the directory
        files_in_directory = os.listdir(directory_path)

        for i,file in enumerate(files_in_directory):
            if file.endswith('.wav'):
                file = file.replace(".wav","")
                r_a = su.read_recording(f"{directory_path}/{file}",SAMPLE_RATE)
                process_recorded_audio(r_a,i,f"{file}",clean_audio_directory)
                # process_audio(i,is_livestream)
                # time.sleep(CHUNK_SIZE_TIME)

        # file = "play_music_29"
        # r_a = su.read_recording(f"{directory_path}/{file}",SAMPLE_RATE)
        # process_recorded_audio(r_a,i,f"{file}")



