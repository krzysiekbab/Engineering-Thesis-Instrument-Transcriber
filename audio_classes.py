import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from playsound import playsound
import os
from constants import *


class Audio:
    """
    Class representing Audio file with basic methods
    """

    def __init__(self, audio_path: str, audio_name: str):
        self.audio_path = audio_path
        self.audio_name = audio_name
        self.audio_data = None
        self.sr = None
        self.load_audio()

    def load_audio(self):
        self.audio_data, self.sr = librosa.load(path=self.audio_path, sr=SAMPLE_RATE)

    def plot_wave(self):
        plt.figure()
        librosa.display.waveshow(self.audio_data, alpha=0.5)
        plt.title(f"Wave plot of {self.audio_path}")
        plt.ylim(-1, 1)
        plt.show()

    def play_audio(self):
        print(f"Playing {self.audio_name} ...")
        playsound(self.audio_path)


class ProcessAudio:
    def __init__(self, audio_file: Audio):
        # audio_data zamiast wskazywać na plik z danych wskazuje na całą klasę Audio
        self.audio_file = audio_file

    def detect_onsets(self, units='samples'):
        print(type(self.audio_file.audio_data))
        o_env = librosa.onset.onset_strength(y=self.audio_file.audio_data, sr=SAMPLE_RATE)
        onsets = librosa.onset.onset_detect(onset_envelope=o_env, sr=SAMPLE_RATE, units=units)
        return onsets

    def divide_into_onset_frames(self, onsets):

        framed_signal = []

        # Adding first and last indexes to an array
        extended_onset_list = np.insert(onsets, 0, 0)
        extended_onset_list = np.insert(extended_onset_list, len(extended_onset_list),
                                        self.audio_file.audio_data.shape[0])
        for i in range(len(extended_onset_list) - 1):
            framed_signal.append(
                self.audio_file.audio_data[extended_onset_list[i]:extended_onset_list[i + 1] - 1])
        return np.array(framed_signal)

    # static method umożliwia wywoływanie metod przez szablon klasy a nie przez instację klasy (np. ProcessAudio.divide_frame_into_smaller_frames)

    @staticmethod
    def divide_onset_frame_into_smaller_frames(data_frame, window_size=WINDOW_SIZE, hop_length=HOP_LENGTH):
        # TODO zmienić to na lepszy sposób dopasowania
        if len(data_frame) < window_size:
            window_size = int(window_size / 4)
            hop_length = int(hop_length / 4)

        smaller_frames = librosa.util.frame(data_frame, frame_length=window_size, hop_length=hop_length, axis=0)
        # multiplying frame by Hanning window
        windowed_frames = np.hanning(window_size) * smaller_frames

        return windowed_frames

    @staticmethod
    def calc_cepstrum(data):
        """
        Calculates the complex cepstrum of a real sequence.
        """
        spectrum = np.fft.fft(data)
        log_spectrum = np.log(np.abs(spectrum))
        cepstrum = np.fft.ifft(log_spectrum).real
        return cepstrum

    def find_f0_frequency(self, data_frame, sr=SAMPLE_RATE, freq_range=FREQUENCY_RANGE):
        cepstrum = self.calc_cepstrum(data_frame)
        """
        Finding fundamental frequency of tested sound using cepstral analysis
        """

        min_freq, max_freq = freq_range
        start = int(sr / max_freq)
        end = int(sr / min_freq)
        narrowed_cepstrum = cepstrum[start:end]
        peak_ix = narrowed_cepstrum.argmax()
        freq0 = sr / (start + peak_ix)

        if freq0 < min_freq or freq0 > max_freq:
            return 0

        return freq0


