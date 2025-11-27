import warnings
warnings.filterwarnings("ignore")

import sounddevice as sd
import numpy as np
import librosa
import os
import threading
import subprocess
from gpiozero import Button, LED
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import queue
import time

# -----------------------------
# CONFIG
# -----------------------------
SAMPLE_RATE = 16000
DOWNSAMPLE_RATE = 8000
BLOCK_DURATION = 0.7
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION)

# -----------------------------
# PHRASE RESPONSES
# -----------------------------
phrases = {
    "phrase1": "Alexa, play classic country",
    "phrase2": "Alexa, next",
    "phrase3": "Alexa, stop music",
    "phrase4": "Alexa, volume up",
    "phrase5": "Alexa, volume down",
}

# -----------------------------
# LOAD REFERENCE AUDIO MFCC
# -----------------------------
def load_reference_mfccs(directory="phrases"):
    references = {}
    for file in os.listdir(directory):
        if file.endswith(".wav"):
            name = os.path.splitext(file)[0]
            path = os.path.join(directory, file)
            audio, sr = librosa.load(path, sr=SAMPLE_RATE)
            audio = librosa.resample(audio, orig_sr=SAMPLE_RATE, target_sr=DOWNSAMPLE_RATE)
            mfcc = librosa.feature.mfcc(y=audio, sr=DOWNSAMPLE_RATE, n_mfcc=13)
            references[name] = mfcc.T
    return references

reference_mfccs = load_reference_mfccs()

# -----------------------------
# RING BUFFER
# -----------------------------
class RingBuffer:
    def __init__(self, size=BLOCK_SIZE):
        self.size = size
        self.buffer = np.zeros(size, dtype=np.float32)
        self.index = 0

    def add(self, data):
        data_len = len(data)
        if data_len >= self.size:
            self.buffer[:] = data[-self.size:]
            return
        end = self.index + data_len
        if end <= self.size:
            self.buffer[self.index:end] = data
        else:
            split = self.size - self.index
            self.buffer[self.index:] = data[:split]
            self.buffer[: data_len - split] = data[split:]
        self.index = (self.index + data_len) % self.size

    def get(self):
        return self.buffer.copy()

ring_buffer = RingBuffer()
audio_queue = queue.Queue()

# -----------------------------
# AUDIO CALLBACK
# -----------------------------
def audio_callback(indata, frames, time_info, status):
    audio_queue.put(indata[:, 0].copy())

# -----------------------------
# MFCC EXTRACTION
# -----------------------------
def extract_mfcc(block):
    block = librosa.resample(block, orig_sr=SAMPLE_RATE, target_sr=DOWNSAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=block, sr=DOWNSAMPLE_RATE, n_mfcc=13)
    return mfcc.T

# -----------------------------
# MATCHING
# -----------------------------
def best_match(mfcc_block):
    scores = {}
    for name, ref_mfcc in reference_mfccs.items():
        distance, _ = fastdtw(mfcc_block, ref_mfcc, dist=euclidean)
        scores[name] = distance
    return min(scores, key=scores.get)

# -----------------------------
# GPIO SETUP
# -----------------------------
button = Button(21)
led = LED(20)

# -----------------------------
# MAIN LOOP
# -----------------------------
def process_audio():
    while True:
        data = audio_queue.get()
        ring_buffer.add(data)
        block = ring_buffer.get()
        mfcc_block = extract_mfcc(block)
        prediction = best_match(mfcc_block)
        print(f"Prediction: {prediction} → {phrases.get(prediction, 'UNKNOWN')}")


def main():
    print("Starting real-time audio matcher...")
    threading.Thread(target=process_audio, daemon=True).start()

    with sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE):
        print("Listening...")
        while True:
            time.sleep(0.1)


if __name__ == "__main__":
    main()
