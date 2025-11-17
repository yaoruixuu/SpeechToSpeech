import sounddevice as sd
import numpy as np
import wave
from gpiozero import Button
from signal import pause
import os
import librosa
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


# --- 1. Extract MFCC from raw audio array ---
def extract_mfcc_from_array(audio_data, sr=16000, n_mfcc=13):
    # Convert int16 to float32 in [-1, 1]
    y = audio_data.astype(np.float32) / 32768.0
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T  # shape: (time_steps, n_mfcc)

# --- 2. DTW distance ---
def dtw_distance(mfcc1, mfcc2):
    distance, path = fastdtw(mfcc1, mfcc2, dist=euclidean)
    return distance

# --- 3. Load template database ---
template_dir = "templates"  # folders: templates/phrase0/*.wav
templates = {}

def load_templates(sr=16000):
    for phrase in os.listdir(template_dir):
        phrase_path = os.path.join(template_dir, phrase)
        if not os.path.isdir(phrase_path):
            continue
        templates[phrase] = []
        for f in os.listdir(phrase_path):
            if f.endswith(".m4a") or f.endswith(".wav"):
                y, _ = librosa.load(os.path.join(phrase_path, f), sr=sr)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                templates[phrase].append(mfcc.T)

# --- 4. Classify a recorded audio ---
def classify(audio_data, sr=16000):
    mfcc_test = extract_mfcc_from_array(audio_data, sr=sr)
    best_phrase = None
    best_distance = float('inf')

    for phrase, mfcc_list in templates.items():
        for mfcc_template in mfcc_list:
            dist = dtw_distance(mfcc_test, mfcc_template)
            if dist < best_distance:
                best_distance = dist
                best_phrase = phrase

    return best_phrase, best_distance

# --- 5. Record audio from USB mic ---
def record_audio(duration=5, sample_rate=16000, channels=1, mic_index=0):
    print("Recording...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, device=mic_index)
    sd.wait()
    print("Recording complete")
    audio_data = np.int16(audio_data * 32767)
    return audio_data

# --- 6. Button callback ---
def on_button_pressed():
    audio_data = record_audio(duration=5, sample_rate=16000, channels=1, mic_index=0)
    predicted_phrase, dist = classify(audio_data, sr=16000)
    print(f"Predicted phrase: {predicted_phrase}, DTW distance: {dist}")


def main():
    load_templates(sr=16000)
    button = Button(17)  # GPIO pin 17
    button.when_pressed = on_button_pressed
    print("Waiting for button press...")
    pause()



if __name__ == "__main__":
    main()