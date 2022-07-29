import numpy as np
import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from itertools import cycle
import os
import pandas as pd


# sns.set_theme(style="white", palette=None)
# color_pa = plt.rcParams["axes.prop_cycle"].by_key()["color"]
# color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])


class Audio:
    def __init__(self, audio_name: str, audio_data: np.ndarray, sr: int):
        self.audio_name = audio_name
        self.audio_data = audio_data
        self.sr = sr

    # def create_pd_series(self):
    #     return pd.Series(data=self.audio_data)

    def fourier_transform(self):
        return np.abs(librosa.stft(self.audio_data))

    def amplitude_to_db_transform(self):
        return librosa.amplitude_to_db(self.fourier_transform())

    def plot_spectrogram(self):
        fig, ax = plt.subplots()
        img = librosa.display.specshow(self.amplitude_to_db_transform(), x_axis='time', y_axis='log', ax=ax)
        ax.set_title(f"Power spectrogram\nFile:{self.audio_name}")
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        plt.show()

    def plot_mel_spectrogram(self):
        s = librosa.feature.melspectrogram(y=self.audio_data, sr=self.sr, n_mels=128)
        s_db_mel = librosa.amplitude_to_db(s, ref=np.max)

        fig, ax = plt.subplots(figsize=(15, 5))
        img = librosa.display.specshow(s_db_mel, x_axis='time', y_axis='log', ax=ax)
        ax.set_title(f"Mel Spectrogram\nFile: {self.audio_name}")
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        plt.show()


def main():
    audio_files = glob('Audio/*')
    audio_path = audio_files[6]
    audio_data, sr = librosa.load(audio_path)  # y - raw data of audio file, sr - sample rate of audio file
    audio_name = audio_files[6][6:]
    print(f'Row data of audio file: {audio_name}')
    print(f'Sample rate (numer of samples in 1s): {sr}')
    print(f'Shape of audio data: {audio_data.shape}')

    # Creating pandas dataset from audio data
    pd.Series(data=audio_data).plot(figsize=(10, 5), lw=1, title=audio_name)
    plt.show()

    # Creating Audio class
    audio_track = Audio(audio_name=audio_name, audio_data=audio_data, sr=sr)
    audio_track_stft = audio_track.fourier_transform()
    audio_track_amp_to_db = audio_track.amplitude_to_db_transform()

    # Spectrogram plots
    audio_track.plot_spectrogram()
    audio_track.plot_mel_spectrogram()


if __name__ == "__main__":
    main()

