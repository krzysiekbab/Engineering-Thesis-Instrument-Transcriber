import numpy as np
import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
from glob import glob
from playsound import playsound

audio_files = glob('Audio/*')
print(audio_files)
audio_path = audio_files[8]
audio_data, sr = librosa.load(audio_path)  # y - raw data of audio file, sr - sample rate of audio file
audio_name = audio_path[6:]
print(f'Row data of audio file: {audio_name}')
print(f'Sample rate (numer of samples in 1s): {sr}')
print(f'Shape of audio data: {audio_data.shape}')

print(f"Playing {audio_name} ...")
playsound(f'Audio/{audio_name}')

# Program steps
# 1. Calculate power spectrum
X = np.fft.fft(audio_data)
X_mag = np.abs(X)

# TODO 2. Find onsets on the signal - main task
