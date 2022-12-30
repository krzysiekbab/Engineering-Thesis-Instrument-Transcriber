import os
# from constants import *
from music21 import lily

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
    # audio_name = '\Audio\\Autumn leaves - saksofon.wav'
    audio_name = '\Audio\\sax\\single notes\\sax F#2.wav'

    # audio_name = '\Audio\\Autumn leaves - pianino.wav'
    # audio_name = '\Audio\Recorded\\gama c dur wokal.wav'
    # audio_name = '\Audio\Recorded\\lala.wav'

    audio_path = directory + audio_name
    audio_name = audio_name.split('\\')[-1][:-4]

    WINDOW_SIZE = 1024
    HOP_SIZE = 512
    TOP_DB = 30
    FREQUENCY_RANGE = (50, 2500)
    SAMPLE_RATE = 44100
    SHORTEST_NOTE = '1/8'

    # CREATE INSTANCE OF "Audio" CLASS
    audio_file = Audio(audio_path=audio_path, audio_name=audio_name, sample_rate=SAMPLE_RATE, hop_size=HOP_SIZE,
                       window_size=WINDOW_SIZE, top_db=TOP_DB, frequency_range=FREQUENCY_RANGE,
                       shortest_note=SHORTEST_NOTE)

    # FILTERING AUDIO SIGNAL
    audio_file.amplitude_filter()
    audio_file.highpass_filter(freq=100)

    # DETECTING ONSETS IN AUDIO FILE
    onsets = audio_file.detect_onsets(aggregate=np.mean)
    # FILTERING ONSETS
    audio_file.filter_onsets()

    # ESTIMATE TEMPO
    tempo = audio_file.estimate_tempo()
    tempo = float(tempo)
    # print(f"Tempo: {tempo}")
    tempo = 150

    # DIVIDE SIGNAL INTO FRAMES FROM ONSET TO ONSET;
    onset_frames = audio_file.divide_into_onset_frames()
    notes_duration_idx, rests_duration_idx = audio_file.find_durations(onset_frames=onset_frames
                                                                       )
    # PLOT TIME SLOTS WERE SOUND WERE IDENTIFIED
    audio_file.plot_notes_ranges(notes_durations=notes_duration_idx)
    # PLOT DETECTED ONSETS
    audio_file.plot_detected_onsets()

    # FIND F0 FREQUENCY
    note_frames = []
    for note_duration in notes_duration_idx:
        first_idx, second_idx = note_duration
        note = audio_file.audio_data[first_idx:second_idx]
        note_frames.append(note)

    found_frequencies = audio_file.find_frequencies(note_frames=note_frames)
    found_frequencies = found_frequencies.round(3)
    # print(found_frequencies)

    # print("len(onset_frames): ", len(onset_frames))
    # print("len(notes_duration_idx): ", len(notes_duration_idx))

    #  CHANGE FREQUENCY TO MUSIC NAMES
    found_notes = change_from_frequency_to_music_notation(found_frequencies)

    # CREATE LENGTH ARRAYS IN SAMPLE AND TIME UNITS
    note_duration_samples, _, notes_duration_times, silences_duration_times = create_duration_tables(notes_duration_idx,
                                                                                                     rests_duration_idx)
    # PICK THE SHORTEST NOTE DURATION IN MUSIC21;
    shortest_note = '1/8'

    # SCALING LENGTHS BY TEMPO
    new_notes_duration_times = []
    for note_time in notes_duration_times:
        new_note_time = time_to_beat(note_time, tempo)
        new_notes_duration_times.append(new_note_time)
    # print("new_notes_duration_times:\n", new_notes_duration_times)

    new_rests_duration_times = []
    for rest_time in silences_duration_times:
        new_rest_time = time_to_beat(rest_time, tempo)
        new_rests_duration_times.append(new_rest_time)
    # print("new_rests_duration_times:\n", new_rests_duration_times)

    # LIST TO NOTES AND RESTS
    elements = []
    for i in range(len(found_notes)):
        if new_notes_duration_times[i] > 0.0:
            elements.append(tuple([found_notes[i], new_notes_duration_times[i]]))
        if new_rests_duration_times[i] > 0.0:
            elements.append(tuple(["rest", new_rests_duration_times[i]]))

    # ROUND NOTE & SILENCE DURATIONS TO MUSIC21 FORMAT; USE THE SHORTEST NOTE DURATION
    elements_music21 = change_to_music21_format(duration_times=elements,
                                                shortest_note=shortest_note)
    print(f"number of notes: {len(new_notes_duration_times)}")
    print(f"number of elements: {len(elements_music21)}")
    # ADD REST ON 1ST INDEX WHEN AUDIO TRACK IS ONE OF THE LISTED BELOW
    if audio_name == 'Autumn leaves - saksofon' or audio_name == 'Autumn leaves - pianino':
        elements_music21.insert(0, ('rest', 1.0))

    values = 0
    for elem in elements_music21:
        values += elem[1]
    print(f"elements_music21:\n{elements_music21}")

    # CHECK DIVISION BY 4 (IF METRUM = 4/4) -> DELETING REDUNDANT BAR IN THE END
    if not (values % 4 == 0) and elements_music21[-1][0] == 'rest':
        division_rest = values % 4
        rest_value = elements_music21[-1][1]
        elements_music21.pop(-1)
        elements_music21.append(('rest', rest_value - division_rest))

    # CREATE STREAM
    stream1 = music21.stream.Stream()

    # ADD TO STREAM ALL THE NOTES AND RESTS
    create_notes_and_rests(elements_music21, stream=stream1)
    stream1.insert(0, music21.metadata.Metadata())
    stream1.metadata.title = audio_name
    stream1.metadata.composer = f'top_db = {audio_file.top_db} [dB]\n' \
                                f'shortest_note = {shortest_note}\n' \
                                f'windows_size = {audio_file.window_size}\n' \
                                f'hop_size = {audio_file.hop_size}\n' \
                                f'sample_rate = {audio_file.sample_rate}\n' \
                                f'frequency_range = {audio_file.frequency_range}'

    # PRINT STREAM IN MUSESCORE
    # stream1.show()
    # pdf_file = music21.midi.translate.streamToPdfFile

    audio_file.plot_wave()
    audio_file.plot_magnitude_spectrum()
    # audio_file.plot_spectrogram()
    # audio_file.plot_mel_spectrogram()

    # WRITE SHEETMUSIC TO PDF WITHOUT OPENING MUSESCORE
    stream1.write('musicxml.pdf', directory + '\Results\PDF\\' + audio_file.audio_name)
    # EXPORT TO MIDI
    # mf = music21.midi.translate.streamToMidiFile(stream1)
    # mf.open(directory + '\Results\midi.mid', 'wb')
    # mf.write()
    # mf.close()


if __name__ == "__main__":
    main()
