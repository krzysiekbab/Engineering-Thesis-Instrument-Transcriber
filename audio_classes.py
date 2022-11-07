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
        o_env = librosa.onset.onset_strength(y=self.audio_file.audio_data, sr=SAMPLE_RATE)
        onsets = librosa.onset.onset_detect(onset_envelope=o_env, sr=SAMPLE_RATE, units=units)
        return onsets

    def divide_into_onset_frames(self, onsets):

        framed_signal = []
        for i in range(len(onsets)):
            if i == len(onsets) - 1:
                framed_signal.append(self.audio_file.audio_data[onsets[i]:self.audio_file.audio_data.shape[0]])
            else:
                framed_signal.append(self.audio_file.audio_data[onsets[i]:onsets[i + 1]])
        return np.array(framed_signal, dtype=object)

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

    def find_frequencies(self, notes_durations):
        found_frequencies = []
        for i in range(len(notes_durations)):
            first_idx, second_idx = notes_durations[i]

            # dividing every onset_frame into smaller frames
            frames = self.divide_onset_frame_into_smaller_frames(self.audio_file.audio_data[first_idx:second_idx])

            f0_candidate = []
            for frame in frames:
                f0_freq = self.find_f0_frequency(frame)
                f0_candidate.append(f0_freq)
            f0_freq = np.median(f0_candidate)
            found_frequencies.append(f0_freq)
        return np.array(found_frequencies)

    @staticmethod
    def find_sound_and_silence_ranges(onset_frames, indexes, top_db):
        note_duration = []
        silence_duration = []
        for i in range(len(onset_frames)):
            onset_frame_splitted = librosa.effects.split(onset_frames[i], top_db=top_db) + indexes[i + 1]

            sound = onset_frame_splitted[0]  # omijamy przypadek gdy wykrywa drugą wartość zwykle przed końcem ramki.
            # Rozwiązuje to problemwzrastających wartości na końcu ramki

            note_duration.append(sound)
            silence = np.array([onset_frame_splitted[0][1], indexes[i + 2]])
            silence_duration.append(silence)
        return note_duration, silence_duration
