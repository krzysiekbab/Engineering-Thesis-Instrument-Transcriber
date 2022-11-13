import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from playsound import playsound
from constants import *
import matplotlib.collections as collections
import os


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

    def play_audio(self):
        print(f"Playing {self.audio_name} ...")
        playsound(self.audio_path)

    def detect_onsets(self, units='samples'):
        o_env = librosa.onset.onset_strength(y=self.audio_data, sr=SAMPLE_RATE)
        onsets = librosa.onset.onset_detect(onset_envelope=o_env, sr=SAMPLE_RATE, units=units)
        return onsets

    def divide_into_onset_frames(self, onsets):

        framed_signal = []
        for i in range(len(onsets)):
            if i == len(onsets) - 1:
                framed_signal.append(self.audio_data[onsets[i]:self.audio_data.shape[0]])
            else:
                framed_signal.append(self.audio_data[onsets[i]:onsets[i + 1]])
        return framed_signal

    def find_sound_and_silence_ranges(self, onset_frames, onsets, top_db):
        note_duration = []
        silence_duration = []
        indexes = np.insert(onsets, 0, 0)
        indexes = np.insert(indexes, len(indexes), self.audio_data.shape[0])
        for i in range(len(onset_frames)):
            onset_frame_splitted = librosa.effects.split(onset_frames[i], top_db=top_db) + indexes[i + 1]

            sound = onset_frame_splitted[0]  # omijamy przypadek gdy wykrywa drugą wartość zwykle przed końcem ramki.
            # Rozwiązuje to problemwzrastających wartości na końcu ramki

            note_duration.append(sound)
            silence = np.array([onset_frame_splitted[0][1], indexes[i + 2]])
            silence_duration.append(silence)
        return note_duration, silence_duration

    @staticmethod
    def divide_onset_frame_into_smaller_frames(data_frame, window_size=WINDOW_SIZE, hop_length=HOP_LENGTH):
        # TODO zmienić to na lepszy sposób dopasowania
        if len(data_frame) <= window_size:
            window_size = int(window_size / 4)
            # hop_length = int(hop_length / 4)
        if len(data_frame) <= 2*window_size:
            window_size = int(window_size / 2)
            # hop_length = int(hop_length / 2)

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
            frames = self.divide_onset_frame_into_smaller_frames(self.audio_data[first_idx:second_idx])
            f0_candidate = []
            for frame in frames:
                f0_freq = self.find_f0_frequency(frame)
                f0_candidate.append(f0_freq)
            # print(f"Frame[{i}], len={len(f0_candidate)}", np.sort(f0_candidate))
            f0_freq = np.median(f0_candidate)
            found_frequencies.append(f0_freq)
        return np.array(found_frequencies)

    def plot_wave(self):
        plt.figure()
        librosa.display.waveshow(self.audio_data, alpha=0.5, sr=SAMPLE_RATE)
        plt.title(f"Wave plot of {self.audio_path}")
        plt.show()

    def plot_magnitude_spectrum(self, f_ratio=0.1):
        X = np.fft.fft(self.audio_data)
        X_mag = np.abs(X)
        plt.figure(figsize=(10, 8))

        f = np.linspace(0, self.sr, len(X_mag))
        f_bins = int(len(X_mag) * f_ratio)
        print(np.argmax(f[:f_bins]))
        plt.plot(f[:f_bins], X_mag[:f_bins])
        plt.xlabel('Częstotliwość (Hz)')
        plt.title(self.audio_name)
        plt.show()

    def plot_spectrogram(self):
        stft_data = np.abs(librosa.stft(self.audio_data))
        stft_data_db = librosa.amplitude_to_db(stft_data)
        fig, ax = plt.subplots()
        img = librosa.display.specshow(stft_data_db, x_axis='time', y_axis='log', ax=ax)
        ax.set_title(f"Spectrogram: {self.audio_name}")
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        plt.show()

    def plot_mel_spectrogram(self):
        s = librosa.feature.melspectrogram(y=self.audio_data, sr=self.sr, n_mels=128)
        s_db_mel = librosa.amplitude_to_db(s, ref=np.max)

        fig, ax = plt.subplots()
        img = librosa.display.specshow(s_db_mel, x_axis='time', y_axis='log', ax=ax)
        ax.set_title(f"Mel Spectrogram: {self.audio_name}")
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        plt.show()

    def plot_notes_ranges(self, notes_durations, top_db):

        x = np.arange(0, self.audio_data.shape[0])/SAMPLE_RATE
        zeros = np.zeros(self.audio_data.shape[0])
        fig, ax = plt.subplots()

        for sound in notes_durations:
            first_idx, second_idx = sound
            zeros[first_idx:second_idx] = 1
        collection = collections.BrokenBarHCollection.span_where(
            x, ymin=0, ymax=np.abs(self.audio_data).max(),
            where=zeros > 0, facecolor='orange',
            label='Obszar występowania dźwięku')
        ax.add_collection(collection)
        librosa.display.waveshow(self.audio_data, sr=SAMPLE_RATE)
        # ax.plot(self.audio_data, color="#ffa500")
        # ax.plot(self)
        ax.set_xlabel('Czas [s]')
        ax.set_ylabel('Amplituda')
        ax.set_title(f'Przedziały występowania dźwięków (threshold = {top_db})')
        ax.legend(loc='lower right')
        plt.show()


if __name__ == "__main__":
    pass
