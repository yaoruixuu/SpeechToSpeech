import warnings
warnings.filterwarnings("ignore")

import sounddevice as sd
import numpy as np
import librosa
import os
import threading
import subprocess
from gpiozero import Button
from signal import pause
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import queue
import time

# -----------------------------
# PHRASE RESPONSES
# -----------------------------
phrases = {
    "phrase1": "Alexa, play classic country",
    "phrase2": "Alexa, play tsn 1200",
    "phrase3": "Alexa, please remind me to call Elain and Al this afternoon",
    "phrase4": "Alexa, remind me to watch the redblacks game",
    "phrase5": "Alexa, remind me to pick up coke zero"
}

# -----------------------------
# LOAD TEMPLATES (16 kHz)
# -----------------------------
templates = {}
template_dir = "templates"

def load_templates(sr=16000, n_mfcc=13):
    for phrase in os.listdir(template_dir):
        path = os.path.join(template_dir, phrase)
        if not os.path.isdir(path):
            continue
        templates[phrase] = []
        for f in os.listdir(path):
            if f.endswith(".wav") or f.endswith(".m4a"):
                y, _ = librosa.load(os.path.join(path, f), sr=sr)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
                mfcc -= np.mean(mfcc, axis=1, keepdims=True)
                mfcc /= np.std(mfcc, axis=1, keepdims=True) + 1e-6
                templates[phrase].append(mfcc.T)

# -----------------------------
# RECORDING (non-blocking, 10s)
# -----------------------------
class Recorder:
    def __init__(self, sample_rate=16000, duration=10):
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * duration)
        self.audio_buffer = []
        self.lock = threading.Lock()
        self.done = threading.Event()

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(status)
        self.audio_buffer.extend(indata[:, 0])
        if len(self.audio_buffer) >= self.max_samples:
            self.done.set()

    def record(self):
        with self.lock:
            self.audio_buffer = []
            self.done.clear()
            with sd.InputStream(samplerate=self.sample_rate, channels=1, callback=self.audio_callback):
                print(f"Recording for {self.max_samples / self.sample_rate:.1f} seconds...")
                self.done.wait()  # wait until buffer is full
            audio_data = np.array(self.audio_buffer[:self.max_samples]).astype(np.float32)
            audio_16k = librosa.resample(audio_data, orig_sr=self.sample_rate, target_sr=16000)
            audio_16k = np.int16(audio_16k * 32767)
            print("Recording complete.")
            return audio_16k

# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
def extract_mfcc(audio_data, sr=16000, n_mfcc=13):
    y = audio_data.astype(np.float32) / 32768.0
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc -= np.mean(mfcc, axis=1, keepdims=True)
    mfcc /= np.std(mfcc, axis=1, keepdims=True) + 1e-6
    mfcc = mfcc[:, ::2]
    return mfcc.T

# -----------------------------
# DTW CLASSIFICATION
# -----------------------------
def classify(audio_data):
    mfcc_test = extract_mfcc(audio_data)
    best_phrase = None
    best_distance = float("inf")
    for phrase, mfcc_list in templates.items():
        for mfcc_template in mfcc_list:
            dist, path = fastdtw(mfcc_test, mfcc_template, dist=euclidean)
            dist /= len(path)  # normalize
            if dist < best_distance:
                best_distance = dist
                best_phrase = phrase
    return best_phrase, best_distance

# -----------------------------
# TTS
# -----------------------------
tts_lock = threading.Lock()
def speak(text):
    with tts_lock:
        subprocess.run(["espeak", text])

# -----------------------------
# AUDIO PROCESSING THREAD
# -----------------------------
recorder = Recorder(duration=10)

def process_audio():
    audio_data = recorder.record()
    phrase, dist = classify(audio_data)
    print(f"Detected: {phrase} (DTW={dist:.2f})")
    if phrase in phrases:
        speak(phrases[phrase])
    else:
        print("Unknown phrase")

# -----------------------------
# BUTTON CALLBACK
# -----------------------------
def on_button_pressed():
    # Only one recording at a time
    if recorder.lock.locked():
        print("Recording already in progress...")
        return
    threading.Thread(target=process_audio).start()

# -----------------------------
# MAIN
# -----------------------------
def main():
    print("Loading templates...")
    load_templates()
    print("Templates loaded.")
    print("System ready — press button and speak.")

    button = Button(17)
    button.when_pressed = on_button_pressed

    pause()

if __name__ == "__main__":
    main()
