import numpy as np
import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
from glob import glob
from playsound import playsound


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

    def calculate_short_fourier_transform(self):
        return np.abs(librosa.stft(self.audio_data))

    def amplitude_to_db_transform(self):
        return librosa.amplitude_to_db(self.calculate_short_fourier_transform())

    def plot_magnitude_spectrum(self, f_ratio=0.1):
        X = np.fft.fft(self.audio_data)
        X_mag = np.abs(X)
        plt.figure(figsize=(10, 5))

        f = np.linspace(0, self.sr, len(X_mag))
        f_bins = int(len(X_mag) * f_ratio)

        plt.plot(f[:f_bins], X_mag[:f_bins])
        plt.xlabel('Frequency (Hz)')
        plt.title(self.audio_name)
        plt.show()

    def plot_spectrogram(self):
        fig, ax = plt.subplots()
        img = librosa.display.specshow(self.amplitude_to_db_transform(), x_axis='time', y_axis='log', ax=ax)
        ax.set_title(f"Power spectrogram\nFile:{self.audio_name}")
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        plt.show()

    def plot_mel_spectrogram(self):
        s = librosa.feature.melspectrogram(y=self.audio_data, sr=self.sr, n_mels=128)
        s_db_mel = librosa.amplitude_to_db(s, ref=np.max)

        fig, ax = plt.subplots(figsize=(10, 5))
        img = librosa.display.specshow(s_db_mel, x_axis='time', y_axis='log', ax=ax)
        ax.set_title(f"Mel Spectrogram\nFile: {self.audio_name}")
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        plt.show()

    def plot_wave(self):
        plt.figure(figsize=(10, 5))
        librosa.display.waveshow(self.audio_data, alpha=0.5)
        plt.title(f"Waveplot of {self.audio_name}")
        plt.ylim(-1, 1)
        plt.show()

    def play_audio(self):
        print(f"Playing {self.audio_name} ...")
        playsound(f'Audio/{self.audio_name}')
    def find_onsets(self):
        pass

def main():
    audio_files = glob('Audio/*')
    print(audio_files)
    audio_path = audio_files[8]
    audio_data, sr = librosa.load(audio_path)  # y - raw data of audio file, sr - sample rate of audio file
    audio_name = audio_path[6:]
    print(f'Row data of audio file: {audio_name}')
    print(f'Sample rate (numer of samples in 1s): {sr}')
    print(f'Shape of audio data: {audio_data.shape}')

    # Creating pandas dataset from audio data
    # pd.Series(data=audio_data).plot(figsize=(10, 5), lw=1, title=audio_name)
    # plt.show()

    # Creating Audio class
    audio_track = Audio(audio_name=audio_name, audio_data=audio_data, sr=sr)
    # plots

    audio_track.plot_wave()
    audio_track.plot_magnitude_spectrum()
    # audio_track.plot_spectrogram()
    # audio_track.plot_mel_spectrogram()

    # Play audio
    audio_track.play_audio()


if __name__ == "__main__":
    main()
    # LIST OF TODOS:

    # TODO 1. Create GUI
    # TODO 2. Learn how to extract sounds from Spectrogram's
    # TODO 3. Learn how to write sheetmusic in Python
    # TODO 4. Learn how to generate pdfs in Python
    # TODO 5. Learn how to generate .exe file to start the application
