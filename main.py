import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from playsound import playsound
import os
from constants import audio_names
from audio_classes import Audio, ProcessAudio

"""

    Steps of algorithm:
    
    1. Importing audio file
    2. Filtering audio.
    3. Detecting onsets.
    4. Dividing audio into frames from onset to onset (first one from beginning to 1st onset)
    5. Analysing every single frame:
        a) f0 detection
        b) note duration
        c) silence duration
    6. Transforming information into future sheetmusic data.
    7. 

"""


# class Audio:
#     def __init__(self, audio_name: str, audio_data: np.ndarray, sr: int):
#         self.audio_name = audio_name
#         self.audio_data = audio_data
#         self.sr = sr
#     def calculate_short_fourier_transform(self):
#         return np.abs(librosa.stft(self.audio_data))
#
#     def amplitude_to_db_transform(self):
#         return librosa.amplitude_to_db(self.calculate_short_fourier_transform())
#
#     def plot_magnitude_spectrum(self, f_ratio=0.1):
#         X = np.fft.fft(self.audio_data)
#         X_mag = np.abs(X)
#         plt.figure(figsize=(10, 5))
#
#         f = np.linspace(0, self.sr, len(X_mag))
#         f_bins = int(len(X_mag) * f_ratio)
#         print(np.argmax(f[:f_bins]))
#         plt.plot(f[:f_bins], X_mag[:f_bins])
#         plt.xlabel('Frequency (Hz)')
#         plt.title(self.audio_name)
#         plt.show()
#
#     def plot_spectrogram(self):
#         fig, ax = plt.subplots()
#         img = librosa.display.specshow(self.amplitude_to_db_transform(), x_axis='time', y_axis='log', ax=ax)
#         ax.set_title(f"Power spectrogram\nFile:{self.audio_name}")
#         fig.colorbar(img, ax=ax, format="%+2.0f dB")
#         plt.show()
#
#     def plot_mel_spectrogram(self):
#         s = librosa.feature.melspectrogram(y=self.audio_data, sr=self.sr, n_mels=128)
#         s_db_mel = librosa.amplitude_to_db(s, ref=np.max)
#
#         fig, ax = plt.subplots(figsize=(10, 5))
#         img = librosa.display.specshow(s_db_mel, x_axis='time', y_axis='log', ax=ax)
#         ax.set_title(f"Mel Spectrogram\nFile: {self.audio_name}")
#         fig.colorbar(img, ax=ax, format="%+2.0f dB")
#         plt.show()
#
#     def plot_wave(self):
#         plt.figure(figsize=(10, 5))
#         librosa.display.waveshow(self.audio_data, alpha=0.5)
#         plt.title(f"Waveplot of {self.audio_name}")
#         plt.ylim(-1, 1)
#         plt.show()
#
#     def play_audio(self):
#         print(f"Playing {self.audio_name} ...")
#         playsound(f'Audio/{self.audio_name}')


def main():
    # audio_files = glob('Audio/*')

    directory = os.getcwd()
    audio_name = '\Audio\A blues scale.wav'
    audio_path = directory + audio_name
    audio_path = audio_path
    audio_file = Audio(audio_path=audio_path, audio_name=audio_name)

    print(f'Row data of audio file: {audio_file.audio_name}')
    print(f'Sample rate (numer of samples in 1s): {audio_file.sr}')
    print(f'Shape of audio data: {audio_file.audio_data.shape}')

    # Play audio
    # audio_file.play_audio()

    process_audio = ProcessAudio(audio_file=audio_file)
    onsets = process_audio.detect_onsets()
    # dividing signal into frames from onset to onset
    onset_frames = process_audio.divide_into_onset_frames(onsets=onsets)
    found_frequencies = []
    # for count, frame in enumerate(onset_frames):
    #     print(f"frame:{count}, {len(frame)}")
    for onset_frame in onset_frames:
        # dividing every onset_frame into smaller frames
        frames = ProcessAudio.divide_onset_frame_into_smaller_frames(onset_frame)
        f0_candidate = []
        for frame in frames:
            f0_freq = process_audio.find_f0_frequency(frame)
            f0_candidate.append(f0_freq)
        f0_freq = np.median(f0_candidate)
        found_frequencies.append(f0_freq)

    found_frequencies = np.array(found_frequencies)
    found_frequencies = found_frequencies.round(3)
    found_notes = []
    for freq in found_frequencies:
        found_notes.append(librosa.hz_to_note(freq))
    i = 0
    for i in range(len(found_frequencies)):
        print(f"Found frequency: {found_frequencies[i]} [Hz] is equal to {found_notes[i]}")





if __name__ == "__main__":
    main()
    # LIST OF TODOS:

    # TODO 1. Function of finding sound length
    # TODO 2. Function of finding silence
    # TODO 3. librosa.onset.onset_detect - understand the flow
    # TODO 4. librosa.onset.onset_strength - understand the flow
    # TODO 5. Implement auto correlation method e. g. "librosa.yin" (try to write my own code)
    # TODO 6. Write filtering function to use at the beginning of signal processing
    # TODO 7. Create scheme of the whole algorithm
    # TODO 8. Learn how to write sheetmusic in Python
    # TODO 9. Learn how to generate pdfs in Python
    # TODO 10. Learn how to generate .exe file to start the application
    # TODO 11. Create GUI
    # TODO 12. Check working of function hz to note
