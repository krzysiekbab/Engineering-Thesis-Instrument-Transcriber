import numpy as np
import librosa
import librosa.display
import music21


def change_from_frequency_to_music_notation(frequencies):
    found_notes = []
    for freq in frequencies:
        found_note = librosa.hz_to_note(freq)
        if len(found_note) == 3:
            found_note = found_note.replace("♯", "#")
        found_notes.append(found_note)
    return found_notes


def create_duration_tables(notes_duration_idx, silences_duration_idx, sr):
    note_duration_samples = []
    silence_duration_samples = []
    note_duration_times = []
    silence_duration_times = []
    for i in range(len(notes_duration_idx)):
        first_idx, second_idx = notes_duration_idx[i][0], notes_duration_idx[i][1]
        note_length_time = second_idx - first_idx
        note_duration_samples.append(note_length_time)
        note_duration_times.append(librosa.samples_to_time(note_length_time, sr=sr))

        first_idx, second_idx = silences_duration_idx[i][0], silences_duration_idx[i][1]
        silence_duration_time = second_idx - first_idx
        silence_duration_samples.append(silence_duration_time)
        silence_duration_times.append(librosa.samples_to_time(silence_duration_time, sr=sr))

    # Changing type from list to an array
    note_duration_samples = np.array(note_duration_samples)
    silence_duration_samples = np.array(silence_duration_samples)
    note_duration_times = np.array(note_duration_times).round(4)
    silence_duration_times = np.array(silence_duration_times).round(4)
    return note_duration_samples, silence_duration_samples, note_duration_times, silence_duration_times


def choose_shortest_note_length(length):
    durations_music_21 = []
    if length == "1/8":
        durations_music_21 = np.linspace(0, 1, num=3)
    if length == "1/16":
        durations_music_21 = np.linspace(0, 1, num=5)
    if length == "1/32":
        durations_music_21 = np.linspace(0, 1, num=9)
    if length == "1/64":
        durations_music_21 = np.linspace(0, 1, num=17)
    if length == "real":
        pass
    return durations_music_21


def change_to_music21_format(duration_times, shortest_note='1/8'):
    note_lengths = choose_shortest_note_length(length=shortest_note)
    duration_times_music21 = []
    len_diff = 0
    for elem in duration_times:
        first_part, second_part = str(elem[1]).split(".")
        first_part = float(first_part)
        second_part = "0." + second_part
        old_second_part = float(second_part)
        second_part = float(second_part) + len_diff
        index = np.abs(note_lengths - second_part).argmin()
        len_diff = second_part - note_lengths[index]
        second_part = note_lengths[index]
        music21_duration = first_part + second_part
        if music21_duration > 0:
            duration_times_music21.append(tuple([elem[0], music21_duration]))
        if first_part == 0.0 and old_second_part == 0.0:
            len_diff = 0
    return duration_times_music21


def create_notes_and_rests(music_notes, stream):
    for note in music_notes:
        if note[0] == "rest":
            stream.append(music21.note.Rest(note[1]))
        else:
            temp_note = music21.note.Note(note[0])
            temp_note.duration.quarterLength = note[1]
            stream.append(temp_note)
    return stream


def time_to_beat(duration, tempo):
    return tempo * duration / 60
