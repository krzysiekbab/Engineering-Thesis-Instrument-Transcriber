import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from playsound import playsound
import matplotlib.collections as collections
from scipy.signal import butter, sosfilt


class Audio:
    def __init__(self, audio_path: str, audio_name: str, sample_rate: int, hop_size: int, window_size: int,
                 top_db: int, frequency_range: tuple, shortest_note: str):
        self.audio_path = audio_path
        self.audio_name = audio_name
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.window_size = window_size
        self.top_db = top_db
        self.frequency_range = frequency_range
        self.shortest_note = shortest_note

        self.audio_data = None
        self.audio_data_filtered = None
        self.onsets = None
        self.envelope = None
        self.load_audio()

    def load_audio(self):
        self.audio_data, _ = librosa.load(path=self.audio_path, sr=self.sample_rate)

    def highpass_filter(self, freq):
        """

        :param freq:
        :return:
        """
        if self.audio_data_filtered is None:
            self.audio_data_filtered == self.audio_data
        self.audio_data_filtered = sosfilt(
            butter(N=4, Wn=freq, btype='highpass', fs=self.sample_rate / 2, output='sos'),
            self.audio_data_filtered)

    def amplitude_filter(self):
        """
        Zero frames where mean amplitude difference is greater than top_db value
        """

        sound_ranges = librosa.effects.split(self.audio_data, top_db=self.top_db, hop_length=self.hop_size,
                                             frame_length=self.window_size)
        self.audio_data_filtered = np.zeros(len(self.audio_data))
        for sound in sound_ranges:
            start_idx = sound[0]
            end_idx = sound[1]
            self.audio_data_filtered[start_idx:end_idx] = self.audio_data[start_idx:end_idx]

    def filter_onsets(self):
        distance_list = []
        onset_frames = librosa.samples_to_frames(self.onsets, hop_length=self.hop_size)

        amplitudes = self.envelope[onset_frames]
        deleted_frames = []
        """Measuring distance in samples between frames"""
        for i in range(len(onset_frames) - 1):
            distance = onset_frames[i + 1] - onset_frames[i]
            distance_list.append(distance)
        """Deleting onsets which occurs too close to each other"""
        for i in range(len(onset_frames) - 1):
            if distance_list[i] < 10:
                if amplitudes[i + 1] > amplitudes[i]:
                    deleted_frames.append(onset_frames[i])
                else:
                    deleted_frames.append(onset_frames[i + 1])
        new_onsets = np.setdiff1d(onset_frames, deleted_frames)
        removed_indexes = []
        """Deleting onsets which amplitude is less than mean of 3 next samples amplitudes divided by 2"""
        for i in range(1, len(new_onsets) - 2):
            if self.envelope[new_onsets[i]] < np.mean(self.envelope[new_onsets[i - 1:i + 2]]) / 2:
                removed_indexes.append(i)
        new_onsets = librosa.frames_to_samples(new_onsets, hop_length=self.hop_size)
        final_onsets = np.delete(new_onsets, removed_indexes)
        # TODO: Above for loop unused.
        self.onsets = new_onsets

    def detect_onsets(self, units='samples', aggregate=np.median):
        o_env = librosa.onset.onset_strength(y=self.audio_data_filtered, sr=self.sample_rate, aggregate=aggregate,
                                             hop_length=self.hop_size)
        onsets = librosa.onset.onset_detect(onset_envelope=o_env, sr=self.sample_rate, units=units,
                                            hop_length=self.hop_size)
        self.onsets = onsets
        self.envelope = o_env
        return onsets

    def estimate_tempo(self):
        tempo = librosa.beat.tempo(onset_envelope=self.envelope, sr=self.sample_rate)
        return tempo

    def play_audio(self):
        print(f"Playing {self.audio_name} ...")
        playsound(self.audio_path)

    def divide_into_onset_frames(self):
        framed_signal = []
        for i in range(len(self.onsets)):
            if i == len(self.onsets) - 1:
                framed_signal.append(self.audio_data[self.onsets[i]:
                                                     self.audio_data.shape[0]])
            else:
                framed_signal.append(self.audio_data[self.onsets[i]:
                                                     self.onsets[i + 1]])
        return framed_signal

    def find_durations(self, onset_frames):
        note_durations = []
        rest_durations = []
        for i in range(len(onset_frames)):
            split_frame = librosa.effects.split(onset_frames[i], top_db=self.top_db, hop_length=self.hop_size,
                                                frame_length=self.window_size) + self.onsets[i]
            sound = split_frame[0]
            if i == len(onset_frames) - 1:
                silence = np.array([split_frame[0][1], self.audio_data.shape[0]])
            else:
                silence = np.array([split_frame[0][1], self.onsets[i + 1]])

            note_durations.append(sound)
            rest_durations.append(silence)

        return note_durations, rest_durations

    def divide_onset_frame_into_smaller_frames(self, data_frame):
        # TODO zmienić to na lepszy sposób dopasowania
        window_size = self.window_size
        hop_size = self.hop_size
        if len(data_frame) <= self.window_size:
            window_size = int(self.window_size / 4)
            # hop_size = int(self.hop_length / 4)
        if len(data_frame) <= 2 * window_size:
            window_size = int(self.window_size / 2)
            # hop_size = int(self.hop_length / 2)

        smaller_frames = librosa.util.frame(data_frame, frame_length=window_size, hop_length=hop_size, axis=0)
        # multiplying frame by Hanning window
        windowed_frames = np.hanning(window_size) * smaller_frames

        return windowed_frames

    @staticmethod
    def calc_cepstrum(frame):
        """
        Calculates the complex cepstrum of a real sequence.
        """
        spectrum = np.fft.fft(frame)
        log_spectrum = np.log(np.abs(spectrum))
        cepstrum = np.fft.ifft(log_spectrum).real
        return cepstrum

    def find_f0_frequency(self, frame):
        """
        Finding fundamental frequency of tested sound using cepstral analysis
        """
        cepstrum = self.calc_cepstrum(frame)
        min_freq, max_freq = self.frequency_range
        start = int(self.sample_rate / max_freq)
        end = int(self.sample_rate / min_freq)
        narrowed_cepstrum = cepstrum[start:end]
        peak_ix = narrowed_cepstrum.argmax()
        freq0 = self.sample_rate / (start + peak_ix)

        if freq0 < min_freq or freq0 > max_freq:
            return None
        return freq0

    def find_frequencies(self, note_frames):
        found_frequencies = []
        for n_frame in note_frames:
            if len(n_frame) > self.window_size:
                small_frames = librosa.util.frame(n_frame, frame_length=self.window_size,
                                                  hop_length=self.hop_size, axis=0)
                small_frames = np.hanning(self.window_size) * small_frames
            else:
                small_frames = [n_frame]
                small_frames = np.hanning(len(n_frame)) * small_frames
            f0_candidate = []

            for s_frame in small_frames:
                f0_freq = self.find_f0_frequency(s_frame)
                if f0_freq is not None:
                    f0_candidate.append(f0_freq)
            f0_freq = np.median(f0_candidate)
            found_frequencies.append(f0_freq)
        return np.array(found_frequencies)

    def plot_wave(self):
        plt.figure()
        librosa.display.waveshow(self.audio_data, alpha=0.5, sr=self.sample_rate)
        plt.title(f"Wave plot of {self.audio_path}")
        # plt.ylabel('Amplituda')
        plt.show()

    def plot_magnitude_spectrum(self, f_ratio=0.1):
        X = np.fft.fft(self.audio_data_filtered)
        X_mag = np.abs(X)
        plt.figure(figsize=(10, 8))

        f = np.linspace(0, self.sample_rate, len(X_mag))
        f_bins = int(len(X_mag) * f_ratio)
        # print(np.argmax(f[:f_bins]))
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
        s = librosa.feature.melspectrogram(y=self.audio_data, sr=self.sample_rate, n_mels=128)
        s_db_mel = librosa.amplitude_to_db(s, ref=np.max)

        fig, ax = plt.subplots()
        img = librosa.display.specshow(s_db_mel, x_axis='time', y_axis='log', ax=ax)
        ax.set_title(f"Mel Spectrogram: {self.audio_name}")
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        plt.show()

    def plot_notes_ranges(self, notes_durations):

        x = np.arange(0, self.audio_data.shape[0]) / self.sample_rate
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
        librosa.display.waveshow(self.audio_data, sr=self.sample_rate)
        ax.set_xlabel('Czas [s]')
        ax.set_ylabel('Amplituda')
        ax.set_title(f'Przedziały występowania dźwięków (threshold = {self.top_db})')
        ax.legend(loc='lower right')
        plt.show()

    def plot_detected_onsets(self):
        onset_frames = librosa.samples_to_frames(self.onsets, hop_length=self.hop_size)
        times = librosa.times_like(self.envelope, sr=self.sample_rate)

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(times, self.envelope, label='strumień widmowy')
        ax.vlines(times[onset_frames], 0, self.envelope.max(), color='r', alpha=0.9, linestyle='--',
                  label='wykryte nuty')
        ax.legend(loc='upper right')
        ax.set_ylabel("Amplituda")
        ax.set_xlabel("Czas [s]")
        plt.show()


if __name__ == "__main__":
    pass
