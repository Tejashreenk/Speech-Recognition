import numpy as np
import sounddevice as sd
from scipy.io import wavfile

def record_save_audio(time, filename, Fs = 16000):
    # Record audio for 4 seconds
    print(f"Start speaking for {filename}! Recording for {time} seconds...")
    s = sd.rec(int(time * Fs), samplerate=Fs, channels=1, dtype='float64')
    sd.wait()  # Wait until the recording is finished
    # print("Recording finished.")

    # Save the original recording
    wavfile.write(f'{filename}.wav', Fs, s)

def play_recording(filename, Fs=16000):
    # Read and play back the original recording
    print("Playing re-read in PCM recording...")
    s_read = wavfile.read(f'{filename}.wav')[1]
    sd.play(s_read, Fs)
    sd.wait()
    return s_read 

def read_recording(filename, Fs=16000):
    # Read the original recording
    s_read = wavfile.read(f'{filename}.wav')[1]
    return s_read 

def create_recordings(audio_array,directory="",Fs=16000):
    for filename in audio_array:
        print(f"recording for {filename}")
        for i in range(30):
            record_save_audio(3,f"{directory}/{filename}_{i+1}",Fs)

def create_suffled_data(filename, Fs=16000):
    arr = []
    for i in range(30):
        arr.append(read_recording(f"{filename}_{i+1}",Fs))
    random_arr = np.random.shuffle(arr)

directory = "Recordings_3005"
phrases = ["odessa","turn_on_the_lights","turn_off_the_lights","what_time_is_it","play_music","stop_music"]
utterances_per_phrase = 30
Fs=22050
create_recordings(phrases,directory,Fs)
'''
# Simulate audio file names
data = {phrase: [f"{phrase}_{i+1}.wav" for i in range(utterances_per_phrase)] for phrase in phrases}

# Shuffle data for each phrase
for files in data.values():
    np.random.shuffle(files)

# Create folds
folds = {i: {'train': [], 'test': []} for i in range(1, 6)}

for i in range(5):
    for phrase, files in data.items():
        test_files = files[i*4:(i+1)*4]
        train_files = [f for f in files if f not in test_files]
        folds[i+1]['train'].extend(train_files)
        folds[i+1]['test'].extend(test_files)

# Write files
for i in range(1, 6):
    with open(f'train {i}.txt', 'w') as f_train, open(f'validation {i}.txt', 'w') as f_test:
        f_train.write('\n'.join(folds[i]['train']))
        f_test.write('\n'.join(folds[i]['test']))

print("Fold files created.")
'''

