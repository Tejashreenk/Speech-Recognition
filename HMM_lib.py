import numpy as np
import librosa
from my_hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import os
import pickle
import mfcc


# Function to extract MFCC features
def extract_mfcc(audio_path, sample_rate, n_mfcc=13):
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
def train_hmm(models, audio_files, labels, sample_rate, i =  0):
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
        model_path = os.path.join(model_dir, f'{label}_hmm_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f'Model for {label} saved at {model_path}')

# Recognize speech
def recognize(models, audio_path, sample_rate):
    mfcc_features = extract_mfcc_calc(audio_path, sample_rate )
    scores = {label: model.score(mfcc_features) for label, model in models.items()}
    return max(scores, key=scores.get)

def recognize_lib(models, audio_path, sample_rate):
    mfcc_features = extract_mfcc(audio_path, sample_rate )
    scores = {label: model.score(mfcc_features) for label, model in models.items()}
    return max(scores, key=scores.get)

def train_using_all_files():
    audio_files = []
    labels = []
    sample_rate = 22050
    directory = "cleaned_audios_3005"
    filenames = ["play_music","stop_music","turn_off_the_lights","turn_on_the_lights","what_time_is_it","odessa"]
    for filename in filenames:
        for i in range(25):
            audio_files.append(f"{directory}/{filename}_{i+1}.wav")
            labels.append(filename)
    train_hmm(models, audio_files, labels, sample_rate)

def load_models(model_dir='models'):
    models = {}
    for file in os.listdir(model_dir):
        if file.endswith('_hmm_model0206.pkl'):
            label = file.replace('_hmm_model0206.pkl', '')
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

def test_one_audio_sample(audio_path,sample_rate):
    # Recognize a new audio file
    new_audio_path = audio_path#'cleaned_audios_3005/odessa_5.wav'
    recognized_label = recognize(models, new_audio_path, sample_rate)
    print(f'Recognized as: {recognized_label}')

def get_train_files(directory):

    # directory = "OdessaRecordings" 
    train_file_paths = ['train 1.txt','train 1.txt','train 2.txt','train 3.txt','train 4.txt','train 5.txt']

    X_train_fold = []
    y_train_fold = []
    for train_file in train_file_paths:
        with open(f"Folds/{train_file}",'r') as file:
            train_set = file.readlines()
        print(train_file) 
        X_train = []
        y_train = []
        for sample in train_set:
            sample = sample.replace('\n','')
            X_train.append(f'{directory}/{sample}')
            sample = sample.replace('.wav','')
            label = sample.split('_')[-1]
            y_train.append(sample.replace(f'_{str(label)}',''))
        X_train_fold.append(X_train)
        y_train_fold.append(y_train)
    return X_train_fold,y_train_fold

def get_val_error(X_train_fold,y_train_fold,models,directory,sample_rate):
    for i in range(len(X_train_fold)):
        print(f"Fold {i}") 
        X_train = X_train_fold[i]
        y_train = y_train_fold[i]
        correct_train = 0
        wrong_train = 0
        for sample in X_train:
            recognized_label = recognize(models, f'{directory}/{sample}', sample_rate)
            print(recognized_label)
            print(y_train[i])

            if X_train[i]==y_train[i]:
                correct_train += 1
            else:
                wrong_train += 1
            print(f"right: {correct_train}; wrong: {wrong_train}")

def test_trained_model(models,directory,sample_rate):
    val_file_paths = ['validation 1.txt','validation 1.txt','validation 2.txt','validation 3.txt','validation 4.txt','validation 5.txt']

    X_test_fold = []
    y_test_fold = []

    for test_file in val_file_paths:
        with open(f"Folds/{test_file}",'r') as file:
            test_set = file.readlines()
        print(test_file) 
        X_test = []
        y_test = []
        correct_test = 0
        wrong_test = 0

        for sample in test_set:
            sample = sample.replace('\n','')
            recognized_label = recognize(models, f'{directory}/{sample}', sample_rate)
            sample = sample.replace('.wav','')
            X_test.append(recognized_label)
            label = sample.split('_')[-1]
            y_test.append(sample.replace(f'_{str(label)}',''))
            print(recognized_label)
            print(X_test[-1])
            if X_test[-1]==y_test[-1]:
                correct_test += 1
            else:
                wrong_test += 1
            print(f"right: {correct_test}; wrong: {wrong_test}")
        
        y_test_fold.append(y_test)
        X_test_fold.append(X_test)

def train_models_with_audiofiles(X_train_fold,y_train_fold,sample_rate):
    for i in range(len(X_train_fold)):
        print(f"Fold {i}") 
        X_train = X_train_fold[i]
        y_train = y_train_fold[i]
    train_hmm(models, X_train, y_train, sample_rate, i)

'''
use_trained_models = True
# Train models
models = {}

if use_trained_models:
    models = load_models("SavedModels")
else:
    train_using_all_files()

test_trained_model(models)
'''
