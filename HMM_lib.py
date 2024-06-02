import numpy as np
import librosa
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import os
import pickle
import mfcc


# Function to extract MFCC features
def extract_mfcc(audio_path, sample_rate =16000, n_mfcc=13):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc_features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_scaled = StandardScaler().fit_transform(mfcc_features.T)
    return mfcc_scaled

# Function to extract MFCC features
def extract_mfcc_calc(audio_path, sample_rate, n_mfcc=13):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc_features = mfcc.calculate_mfcc_for_onerecording(False, audio_path,sample_rate=sample_rate)
    # mfcc_features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_scaled = StandardScaler().fit_transform(mfcc_features.T)
    return mfcc_scaled

# Train HMM model
def train_hmm(models, audio_files, labels, sample_rate):
    for label in np.unique(labels):
        training_data = [extract_mfcc_calc(audio_files[i], sample_rate) for i in range(len(labels)) if labels[i] == label]
        lengths = [len(x) for x in training_data]
        training_data = np.vstack(training_data)

        model = hmm.GaussianHMM(n_components=3, covariance_type='diag', n_iter=2000)
        model.fit(training_data, lengths)
        models[label] = model

        # Save the model
        model_dir = "SavedModels"
        model_path = os.path.join(model_dir, f'{label}_hmm_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f'Model for {label} saved at {model_path}')

# Recognize speech
def recognize(models, audio_path, sample_rate):
    mfcc_features = extract_mfcc_calc(audio_path, sample_rate )
    scores = {label: model.score(mfcc_features) for label, model in models.items()}
    return max(scores, key=scores.get)

audio_files = []
labels = []
sample_rate = 22050
directory = "cleaned_audios_3005"
filenames = ["play_music","stop_music","turn_off_the_lights","turn_on_the_lights","what_time_is_it","odessa"]
for filename in filenames:
    for i in range(29):
        audio_files.append(f"{directory}/{filename}_{i+1}.wav")
        labels.append(filename)

# Train models
models = {}
train_hmm(models, audio_files, labels, sample_rate)

# Recognize a new audio file
new_audio_path = 'cleaned_audios//what_time_is_it_29.wav'
recognized_label = recognize(models, new_audio_path, sample_rate)
print(f'Recognized as: {recognized_label}')
