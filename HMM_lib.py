import numpy as np
import librosa
from my_hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import os
import pickle
import mfcc


# Function to extract MFCC features
def extract_mfcc(audio_path, sample_rate =22050, n_mfcc=13):
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
    print("Training started ...")
    for label in np.unique(labels):
        training_data = [extract_mfcc_calc(audio_files[i], sample_rate) for i in range(len(labels)) if labels[i] == label]
        lengths = [len(x) for x in training_data]
        training_data = np.vstack(training_data)

        model = hmm.MyGaussianHMM(n_components=3,tol=1e-4, n_iter=2000,implementation="scaling")
        model.fit(training_data, lengths)
        models[label] = model

        # Save the model
        model_dir = "SavedModels"
        model_path = os.path.join(model_dir, f'{label}_hmm_model_0406.pkl')
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

def load_models(model_dir='models'):
    models = {}
    for file in os.listdir(model_dir):
        if file.endswith('_hmm_model_0406.pkl'):
            label = file.replace('_hmm_model_0406.pkl', '')
            model_path = os.path.join(model_dir, file)
            with open(model_path, 'rb') as f:
                # print(f)
                models[label] = pickle.load(f)
                # print(models[label])
            if not models:
                raise ValueError("No models available for recognition.")
            else:
                print(f'Model for {label} loaded from {model_path}')
    return models

use_trained_models = False

if use_trained_models:
    models = load_models("SavedModels")
else:
    train_hmm(models, audio_files, labels, sample_rate)


# Recognize a new audio file
new_audio_path = 'cleaned_audios/odessa_5.wav'
recognized_label = recognize(models, new_audio_path, sample_rate)
print(f'Recognized as: {recognized_label}')
