import librosa
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import os

# --- 1. Load audio and extract MFCCs ---
def extract_mfcc(path, n_mfcc=13, sr=16000):
    y, _ = librosa.load(path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)  # y= instead of positional
    return mfcc.T  # shape: (time_steps, n_mfcc)


# --- 2. DTW distance between two MFCC sequences ---
def dtw_distance(mfcc1, mfcc2):
    distance, path = fastdtw(mfcc1, mfcc2, dist=euclidean)
    return distance

# --- 3. Build template database ---
# Suppose we have folders: templates/phrase0/*.wav, templates/phrase1/*.wav, ...
template_dir = "templates"
templates = {}

for phrase in os.listdir(template_dir):
    phrase_path = os.path.join(template_dir, phrase)
    if not os.path.isdir(phrase_path):
        continue  # skip files like .DS_Store
    templates[phrase] = [
        extract_mfcc(os.path.join(phrase_path, f))
        for f in os.listdir(phrase_path)
        if f.endswith(".m4a")
    ]


# --- 4. Classify a new utterance ---
def classify(audio_path):
    mfcc_test = extract_mfcc(audio_path)
    best_phrase = None
    best_distance = float('inf')
    
    for phrase, mfcc_list in templates.items():
        for mfcc_template in mfcc_list:
            dist = dtw_distance(mfcc_test, mfcc_template)
            if dist < best_distance:
                best_distance = dist
                best_phrase = phrase
    return best_phrase, best_distance

# --- 5. Example usage ---
test_audio = "test.m4a"
predicted_phrase, dist = classify(test_audio)
print(f"Predicted phrase: {predicted_phrase}, DTW distance: {dist}")
