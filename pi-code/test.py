import sounddevice as sd
import numpy as np
import wave

# Set parameters
duration = 5  # seconds to record
sample_rate = 44100  # samples per second
channels = 1  # mono audio

print("Recording...")
# Record audio

usb_mic_index = 0  # Replace with the actual index from query_devices()
audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, device=usb_mic_index)
sd.wait()  # Wait until recording is finished
print("Recording complete")

# Convert to 16-bit PCM format
audio_data = np.int16(audio_data * 32767)

# Save to WAV file
filename = "output.wav"
with wave.open(filename, 'w') as wf:
    wf.setnchannels(channels)
    wf.setsampwidth(2)  # 2 bytes for 16-bit audio
    wf.setframerate(sample_rate)
    wf.writeframes(audio_data.tobytes())

print(f"Audio saved to {filename}")
