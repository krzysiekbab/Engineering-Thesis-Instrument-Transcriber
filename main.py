import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from playsound import playsound
import os
from constants import *
from functions import *
from audio_class import Audio
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


def main():
    # SET PATH OF AUDIO FILE
    directory = os.getcwd()
    # audio_name = '\Audio\piano\A blues scale.wav'
    # audio_name = '\Audio\piano\A blues scale with breaks.wav'
    # audio_name = '\Audio\piano\C ionian scale.wav'
    # audio_name = '\Audio\\piano single notes\piano A4.wav'

    # audio_name = '\Audio\sax\single notes\sax F#2.wav'
    # audio_name = '\Audio\sax\\melody d ionian 120 bpm.wav'

    # audio_name = '\Audio\sax\\autumn leaves high(150 bpm).wav'
    # audio_name = '\Audio\sax\\C ionian scale sax.wav'
    # audio_name = '\Audio\sax\\F# blues scale sax.wav'

    audio_name = '\Audio\Recorded\\output.wav'


    audio_path = directory + audio_name
    audio_path = audio_path

    # CREATE INSTANCE OF "Audio" CLASS
    audio_file = Audio(audio_path=audio_path, audio_name=audio_name)
    # PICK TOP_DB VALUE
    top_db = 40

    # DETECT ONSETS IN AUDIO FILE
    audio_file.filter_audio(top_db=top_db)

    onsets = audio_file.detect_onsets()
    print(f"Found onsets: (len: {len(onsets)})\n", onsets)
    audio_file.filter_onsets()
    onsets = audio_file.onsets
    print(f"Filtered onsets: (len: {len(onsets)})\n", onsets)
    tempo = audio_file.estimate_tempo()

    print(f"Tempo: {tempo}")
    tempo = float(tempo)

    # DIVIDE SIGNAL INTO FRAMES FROM ONSET TO ONSET;
    onset_frames = audio_file.divide_into_onset_frames(onsets=onsets)

    notes_duration_idx, rests_duration_idx = audio_file.find_durations(onset_frames=onset_frames,
                                                                       onsets=onsets,
                                                                       top_db=top_db)
    # PLOT TIME SLOTS WERE SOUND WERE IDENTIFIED
    audio_file.plot_notes_ranges(notes_durations=notes_duration_idx, top_db=top_db)
    #
    # FIND F0 FREQUENCY
    note_frames = []
    for note_duration in notes_duration_idx:
        first_idx, second_idx = note_duration
        note = audio_file.audio_data[first_idx:second_idx]
        note_frames.append(note)

    # print(note_frames)
    found_frequencies = audio_file.find_frequencies(note_frames=note_frames)
    found_frequencies = found_frequencies.round(3)
    # print(found_frequencies)
    # for freq in found_frequencies:
    #     print(freq)

    # print("len(onset_frames): ", len(onset_frames))
    # print("len(notes_duration_idx): ", len(notes_duration_idx))

    # # CHANGE FREQUENCY TO MUSIC NAMES
    found_notes = change_from_frequency_to_music_notation(found_frequencies)
    # TODO: repair when identified frequency out from FREQUENCY_RANGE

    # Printing results
    # for i in range(len(found_frequencies)):
    #     print(f"Frame [{i:2}] (), Found frequency: {found_frequencies[i]:.3f} [Hz] is equal to {found_notes[i]}")

    # # CREATE LENGTH ARRAYS IN SAMPLE AND TIME UNITS
    note_duration_samples, _, notes_duration_times, silences_duration_times = create_duration_tables(notes_duration_idx,
                                                                                                     rests_duration_idx)
    print("notes_duration_times:\n", notes_duration_times)
    # for i in range(len(found_frequencies)):
    # print(
    #     f"Frame [{i:2}] ({note_duration_samples[i]} Samples), Found frequency: {found_frequencies[i]:.3f} [Hz] is "
    #     f"equal to {found_notes[i]}")
    # PICK THE SHORTEST NOTE DURATION IN MUSIC21;
    shortest_note = '1/16'

    # SCALING LENGTHS BY TEMPO
    new_notes_duration_times = []
    for note_time in notes_duration_times:
        new_note_time = time_to_beat(note_time, tempo)
        new_notes_duration_times.append(new_note_time)
    print("new_notes_duration_times:\n", new_notes_duration_times)
    new_rests_duration_times = []
    print(silences_duration_times)

    for rest_time in silences_duration_times:
        new_rest_time = time_to_beat(rest_time, tempo)
        new_rests_duration_times.append(new_rest_time)
    print("new_rests_duration_times:\n", new_rests_duration_times)

    # ROUND NOTE & SILENCE DURATIONS TO MUSIC21 FORMAT; USE THE SHORTEST NOTE DURATION
    notes_duration_times_music21 = change_to_music21_format(duration_times=new_notes_duration_times,
                                                            shortest_note=shortest_note)
    silences_duration_times_music21 = change_to_music21_format(duration_times=new_rests_duration_times,
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
    # pdf_file = music21.midi.translate.streamToPdfFile
    audio_file.plot_wave()
    # audio_file.plot_magnitude_spectrum()
    # audio_file.plot_spectrogram()
    # audio_file.plot_mel_spectrogram()
    # WRITE SHEETMUSIC TO PDF WITHOUT OPENING MUSESCORE
    # stream1.write('musicxml.pdf', input("Enter The Song Name (Add File Extension): "))
    """Exporting to midi file"""
    # mf = music21.midi.translate.streamToMidiFile(stream1)
    # mf.open(directory + '\Results\midi.mid', 'wb')
    # mf.write()
    # mf.close()

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
    # TODO 2. Write descriptions and hits of classes and methods in audio_class.py
    # TODO 5. Implement auto correlation method e. g. "librosa.yin" (try to write my own code)
    # TODO 10. Learn how to generate .exe file to start the application
    # TODO 11. Create GUI
    # TODO 12. Check working of function hz to note by librosa
    # TODO 13. Manage saving pdf-s to "Results" folder.
    # TODO: Filter music notatnion - przyporządkuj wartości rytmiczne do długości każdego onset frame'a.
    #       Zawierające się w nim nuta oraz pauza łącznie muszą dawać taką samą wartość
    # TODO: Audio filtering
    # TODO: Animated plot of audio - if have time
    # TODO: zmienić wszystkie gridy i pack na place
