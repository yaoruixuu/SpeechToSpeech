from gpiozero import Button
import sounddevice as sd

# gpio 2
button = Button(2)
button.wait_for_press()
record()

def record():
    duration = 10
    sd.default.samplerate = fs
    myRecording = sd.record(int(duration * fs), samplerate=fs, channels = 1)
