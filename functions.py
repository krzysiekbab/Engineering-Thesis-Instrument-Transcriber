import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from playsound import playsound
from constants import *
import matplotlib.collections as collections
import music21


def change_from_frequency_to_music_notation(frequencies):
    found_notes = []
    for freq in frequencies:
        found_note = librosa.hz_to_note(freq)
        # Check if is any "♯"
        if len(found_note) == 3:
            found_note = found_note.replace("♯", "#")
        found_notes.append(found_note)
    return found_notes


def create_duration_tables(notes_duration_idx, silences_duration_idx):
    note_duration_samples = []
    silence_duration_samples = []
    note_duration_times = []
    silence_duration_times = []
    for i in range(len(notes_duration_idx)):
        first_idx, second_idx = notes_duration_idx[i][0], notes_duration_idx[i][1]
        note_length_time = second_idx - first_idx
        note_duration_samples.append(note_length_time)
        note_duration_times.append(librosa.samples_to_time(note_length_time, sr=SAMPLE_RATE))

        first_idx, second_idx = silences_duration_idx[i][0], silences_duration_idx[i][1]
        silence_duration_time = second_idx - first_idx
        silence_duration_samples.append(silence_duration_time)
        silence_duration_times.append(librosa.samples_to_time(silence_duration_time, sr=SAMPLE_RATE))

    # Changing type from list to an array
    note_duration_samples = np.array(note_duration_samples)
    silence_duration_samples = np.array(silence_duration_samples)
    note_duration_times = np.array(note_duration_times).round(4)
    silence_duration_times = np.array(silence_duration_times).round(4)
    return note_duration_samples, silence_duration_samples, note_duration_times, silence_duration_times


def choose_shortest_note_length(length):
    durations_music_21 = []

    if length == "1/8":
        durations_music_21 = np.linspace(0, 1, num=3)[1:-1]
    if length == "1/16":
        durations_music_21 = np.linspace(0, 1, num=5)[1:-1]
    if length == "1/32":
        durations_music_21 = np.linspace(0, 1, num=9)[1:-1]
    if length == "1/64":
        durations_music_21 = np.linspace(0, 1, num=17)[1:-1]
    if length == "real":
        pass
    return durations_music_21


def change_to_music21_format(duration_times, shortest_note='1/16'):
    note_lengths = choose_shortest_note_length(length=shortest_note)
    duration_times_music21 = []
    for elem in duration_times:
        first_part, second_part = str(elem).split(".")
        second_part = "0." + second_part
        index = np.abs(note_lengths - float(second_part)).argmin()
        second_part = str(note_lengths[index])[2:]
        music21_duration = float(first_part + "." + second_part)
        duration_times_music21.append(music21_duration)
    return duration_times_music21


def create_notes_and_rests(music_notes, stream):
    for note in music_notes:
        if note[0] == "rest":
            stream.append(music21.note.Rest(note[1]))
        else:
            note1 = music21.note.Note(note[0])
            note1.duration.quarterLength = note[1]
            stream.append(note1)
    return stream


def time_to_beat(duration, tempo):
    return tempo * duration / 60

def record_audio():
    pass
# tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=SAMPLE_RATE)
# print(f"Tempo: {tempo}")
# def time_to_beat(duration, tempo):
#     return (tempo * duration / 60)


# first_frame = audio_file.audio_data[:onsets[0] - 1] - FIRST FRAME OF SILENCE

# print(f"Liczba wykrytych onsetów: {len(onsets)}")
# print(f"Liczba ramek onset_frames: {len(onset_frames)} (równa liczba onsetów)")
# print(f"Liczba indeksów:{len(indexes)} (dodany indeks początkowy i końcowy)\n")

# Printing results
# for i in range(len(found_frequencies)):
#     print(f"Frame [{i:2}], Found frequency: {found_frequencies[i]:.3f} [Hz] is equal to {found_notes[i]}")

# SHOWING RANGES OF SOUNDS
# for i in range(len(onset_frames)):
#     print(
#         f"Ramka[{i}]: ({indexes[i + 1]}:{indexes[i + 2]}),"
#         f"\tPoczątek dźwięku: {onsets[i]},"
#         f"\tDługość dźwięku: {notes_duration_idx[i]},"
#         f"\tDługość ciszy: {silences_duration_idx[i]}")
