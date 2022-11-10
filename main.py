import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from playsound import playsound
import os
from constants import *
from functions import *
from audio_classes import Audio, ProcessAudio
import matplotlib.collections as collections
import music21

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


def main():
    # SET PATH OF AUDIO FILE
    directory = os.getcwd()
    audio_name = '\Audio\A blues scale with breaks.wav'
    # audio_name = '\Audio\A blues scale.wav'
    # audio_name = '\Audio\C ionian scale.wav'
    # audio_name = '\Audio\\piano single notes\piano A4.wav'

    audio_path = directory + audio_name
    audio_path = audio_path

    # CREATE INSTANCE OF "Audio" CLASS
    audio_file = Audio(audio_path=audio_path, audio_name=audio_name)
    onset_env = librosa.onset.onset_strength(y=audio_file.audio_data, sr=SAMPLE_RATE)

    # CREATE INSTANCE OF "ProcessAudio" CLASS
    process_audio = ProcessAudio(audio_file=audio_file)
    onsets = process_audio.detect_onsets()

    # CREATE INDEX ARRAY (ONSETS WITH ADDED INDEXES OF FIRST AND LAST SAMPLE)
    indexes = np.insert(onsets, 0, 0)
    indexes = np.insert(indexes, len(indexes), audio_file.audio_data.shape[0])

    # DIVIDE SIGNAL INTO FRAMES FROM ONSET TO ONSET; RETURNS PARTS OF SIGNAL
    onset_frames = process_audio.divide_into_onset_frames(onsets=onsets)

    # PICK TOP_DB VALUE
    top_db = 30  # TODO rozwiącać problem gdy zwiększamy do 50 dB, i franctional prat ma stosunkowe małe wartości - rozw.: usuwać dźwięki których długość to 0.

    notes_duration_idx, silences_duration_idx = ProcessAudio.find_sound_and_silence_ranges(onset_frames=onset_frames,
                                                                                           indexes=indexes,
                                                                                           top_db=top_db)

    # PLOT TIME SLOTS WERE SOUND WERE IDENTIFIED
    # process_audio.plot_notes_ranges(notes_durations=notes_duration_idx, top_db=top_db)

    # FIND F0 FREQUENCY
    found_frequencies = process_audio.find_frequencies(notes_durations=notes_duration_idx)
    found_frequencies = found_frequencies.round(3)

    # CHANGE FREQUENCY TO MUSIC NAMES
    found_notes = change_from_frequency_to_music_notation(found_frequencies)

    # CREATE LENGTH ARRAYS IN SAMPLE AND TIME UNITS
    _, _, notes_duration_times, silences_duration_times = create_duration_tables(notes_duration_idx,
                                                                                 silences_duration_idx)
    # PICK THE SHORTEST NOTE DURATION IN MUSIC21;
    shortest_note = '1/32'

    # ROUND NOTE & SILENCE DURATIONS TO MUSIC21 FORMAT; USE THE SHORTEST NOTE DURATION
    notes_duration_times_music21 = change_duration_to_music21_format(duration_times=notes_duration_times,
                                                                     shortest_note=shortest_note)
    silences_duration_times_music21 = change_duration_to_music21_format(duration_times=silences_duration_times,
                                                                        shortest_note=shortest_note)
    # CREATE LIST OF TUPLES OF BOTH NOTES AND RESTS AND THEIR DURATIONS
    music_representation = []
    for i in range(len(found_notes)):
        if notes_duration_times[i] != 0:
            music_representation.append(tuple([found_notes[i], notes_duration_times_music21[i]]))
        if silences_duration_times[i] != 0:
            music_representation.append(tuple(["rest", silences_duration_times_music21[i]]))

    print(f"Music representation ready for music21:\n")
    for item in music_representation:
        print(item)

    # CREATE STREAM
    stream1 = music21.stream.Stream()

    # ADD TO STREAM ALL THE NOTES AND RESTS
    create_notes_and_rests(music_representation, stream=stream1)

    # PRINT STREAM IN MUSESCORE
    stream1.show()

    # WRITE SHEETMUSIC TO PDF WITHOUT OPENING MUSESCORE
    # stream1.write('musicxml.pdf', input("Enter The Song Name (Add File Extension): "))

    # stream1.show('text')
    # stream2 = stream1.makeMeasures()
    # stream2.show('text')
    # fp = "D:\Studia\MusicApp\Sheetmusic")
    # GEX = music21.musicxml.m21ToXml.Gener alObjectExporter()
    # m = GEX.fromDiatonicScale(stream1)
    # s = music21.converter.parse(stream1)


if __name__ == "__main__":
    main()
    # LIST OF TODOS:

    # TODO 0: Get rid of Audio class from main.py
    # TODO 1. Write tempo function
    # TODO 2. Write descriptions and hits of classes and methods in audio_classes.py
    # TODO 3. librosa.onset.onset_detect - understand the flow
    # TODO 4. librosa.onset.onset_strength - understand the flow
    # TODO 5. Implement auto correlation method e. g. "librosa.yin" (try to write my own code)
    # TODO 6. Write filtering function to use at the beginning of signal processing
    # TODO 7. Create scheme of the whole algorithm
    # TODO 10. Learn how to generate .exe file to start the application
    # TODO 11. Create GUI
    # TODO 12. Check working of function hz to note by librosa
