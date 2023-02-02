import tkinter as tk
import matplotlib.pyplot as plt
import pygame
import librosa.display
import os
import subprocess
from note_functions import *
import time
import tkinter.ttk as ttk
import threading
import pyaudio
import wave
from PIL import Image, ImageTk
from tkinter import filedialog
from matplotlib import collections
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
from audio_class import Audio

count_pages = 0
path = None
paused = False
stream = None
pdf_path = None
audio_instance = None
notes_durations = None
song_length = 0
creation_time = None
record_frame = True
stopped = False
recording = False
audio = None
recorded_stream = None
frames = []
warning_placed = False
file_chosen = False
algorithm_done = False


def move_next_page():
    global count_pages
    count_pages += 1

    if count_pages == 1:
        mid_page_1.forget()
        bottom_page_1.forget()
        mid_page_2.pack(fill="x")
        bottom_page_2.pack(fill="x", expand=True)
    if count_pages == 2:
        mid_page_2.forget()
        bottom_page_2.forget()
        mid_page_3.pack(fill="x")
        bottom_page_3.pack(fill="x", expand=True)
    if count_pages == 3:
        mid_page_4.pack(fill="x")
        bottom_page_4.pack(fill="x", expand=True)
        mid_page_3.forget()
        bottom_page_3.forget()


def move_previous_page():
    global count_pages
    count_pages -= 1
    if count_pages == 0:
        bottom_page_2.forget()
        mid_page_2.forget()
        mid_page_1.pack(fill="x")
        bottom_page_1.pack(fill="x", expand=True)
    if count_pages == 1:
        bottom_page_3.forget()
        mid_page_3.forget()
        mid_page_2.pack(fill="x")
        bottom_page_2.pack(fill="x", expand=True)
    if count_pages == 2:
        bottom_page_4.forget()
        mid_page_4.forget()
        mid_page_3.pack(fill="x")
        bottom_page_3.pack(fill="x", expand=True)


def select_file():
    global path, audio_name
    directory = os.getcwd() + "\\Audio"
    filetypes = (
        (".wav files", "*.wav"),
    )
    path = filedialog.askopenfilename(
        title='Wybierz plik',
        initialdir=directory,
        filetypes=filetypes)
    audio_name.set(path.split('/')[-1][:-4])
    if path == "":
        path = None
    print(f"Loaded file: {audio_name.get()}, type: {type(path)}")


def check_path():
    global path, file_chosen
    if path is None:
        next_button_page_1["state"] = "disabled"
        file_chosen = False
    else:
        file_chosen = True
        if algorithm_done:
            next_button_page_2["state"] = "disabled"
        switch_button_state(next_button_page_1)


def switch_to_record_frame():
    global record_frame, path
    if path:
        next_button_page_1["state"] = "disabled"
    if record_frame:
        mid_page_1.forget()
        bottom_page_1.forget()
        mid_page_12.pack(fill="x")
        bottom_page_1.pack(fill="x", expand=True)
        record_frame = False

    else:
        mid_page_12.forget()
        bottom_page_1.forget()
        mid_page_1.pack(fill="x")
        bottom_page_1.pack(fill="x", expand=True)
        record_frame = True


def handle_recording():
    global recording

    if recording:
        recording = False
        # stop_recording_button.config(fg="black")
    else:
        recording = True
        # stop_recording_button.config(fg="red")
        threading.Thread(target=start_recording).start()


def start_recording():
    global audio, recorded_stream, frames

    audio = pyaudio.PyAudio()
    recorded_stream = audio.open(format=pyaudio.paInt16, channels=2, rate=44100,
                                 input=True, frames_per_buffer=1024)
    frames = []
    start = time.time()
    switch_button_state(stop_recording_button)
    while recording:
        data = recorded_stream.read(1024)
        frames.append(data)

        passed = time.time() - start
        secs = passed % 60
        mins = passed // 60
        hours = mins // 60
        time_label.config(text=f"{int(hours):02d}:{int(mins):02d}:{int(secs):02d}")
        # , fg = "red"


def stop_recording():
    global audio
    global recorded_stream

    recorded_stream.stop_stream()
    recorded_stream.close()
    audio.terminate()
    switch_button_state(save_recorded_audio)
    switch_button_state(file_name_entry)


def save_recording():
    global audio, frames, warning_placed, path, audio_name
    directory = os.getcwd() + "\\Audio\\Recorded\\"

    filename = file_name_entry.get()
    if filename != "":
        if warning_placed:
            warning["text"] = ""
        # Save the recorded data as a WAV file
        file = directory + filename + ".wav"
        wf = wave.open(file, 'wb')
        wf.setnchannels(2)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b''.join(frames))
        wf.close()

        path = file
        audio_name.set(path.split('\\')[-1][:-4])
        print(f"Loaded file: {audio_name.get()}")

    else:
        warning.place(x=25, y=390, width=500, height=25)
        warning["text"] = "Nazwa pliku nie może być pusta !!!"
        warning_placed = True


def clear_recording_window():
    file_name_entry.delete(0, tk.END)
    time_label["text"] = "00:00:00"
    record_audio_button_2["state"] = "normal"
    file_name_entry["state"] = "disabled"
    save_recorded_audio["state"] = "disabled"


def switch_button_state(button_type):
    if button_type["state"] == "normal":
        if button_type is next_button_page_1 or button_type is next_button_page_2:
            pass
        else:
            button_type["state"] = "disabled"
    else:
        button_type["state"] = "normal"


def run_application(audio_path, audio_name):
    global stream, pdf_path, audio_instance, notes_durations, creation_time, algorithm_done

    WINDOW_SIZE = int(selected_window_size.get())
    HOP_SIZE = int(selected_hop_size.get())
    TOP_DB = int(selected_threshold.get())
    FREQUENCY_RANGE = (int(selected_min_frequency.get()), int(selected_max_frequency.get()))
    SAMPLE_RATE = 44100
    shortest_note = selected_shortest_note.get()
    audio_name = audio_name.get()

    print(f"WINDOW_SIZE: {WINDOW_SIZE}\n"
          f"HOP_SIZE: {HOP_SIZE}\n"
          f"TOP_DB: {TOP_DB}\n"
          f"FREQUENCY_RANGE: {FREQUENCY_RANGE}\n"
          f"SAMPLE_RATE: {SAMPLE_RATE}\n"
          f"shortest_note: {shortest_note}")

    # CREATE INSTANCE OF "Audio" CLASS
    audio_file = Audio(audio_path=audio_path, audio_name=audio_name, sample_rate=SAMPLE_RATE, hop_size=HOP_SIZE,
                       window_size=WINDOW_SIZE, top_db=TOP_DB, frequency_range=FREQUENCY_RANGE,
                       shortest_note=shortest_note)

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

    # DIVIDE SIGNAL INTO FRAMES FROM ONSET TO ONSET;
    onset_frames = audio_file.divide_into_onset_frames()
    notes_duration_idx, rests_duration_idx = audio_file.find_durations(onset_frames=onset_frames)

    # FIND F0 FREQUENCY
    note_frames = []
    for note_duration in notes_duration_idx:
        first_idx, second_idx = note_duration
        note = audio_file.audio_data[first_idx:second_idx]
        note_frames.append(note)

    found_frequencies = audio_file.find_frequencies(note_frames=note_frames)
    found_frequencies = found_frequencies.round(3)

    #  CHANGE FREQUENCY TO MUSIC NAMES
    found_notes = change_from_frequency_to_music_notation(found_frequencies)

    # CREATE LENGTH ARRAYS IN SAMPLE AND TIME UNITS
    note_duration_samples, _, notes_duration_times, silences_duration_times = create_duration_tables(notes_duration_idx,
                                                                                                     rests_duration_idx,
                                                                                                     sr=SAMPLE_RATE)
    # SCALING LENGTHS BY TEMPO
    new_notes_duration_times = []
    for note_time in notes_duration_times:
        new_note_time = time_to_beat(note_time, tempo)
        new_notes_duration_times.append(new_note_time)

    new_rests_duration_times = []
    for rest_time in silences_duration_times:
        new_rest_time = time_to_beat(rest_time, tempo)
        new_rests_duration_times.append(new_rest_time)

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
    # stream1.metadata.composer = ""
    stream = stream1
    data = datetime.now()
    data = data.strftime("%d-%m-%Y %H.%M.%S")
    creation_time = f" ({data})"
    pdf_path = os.getcwd() + '\Results\PDF\\' + audio_file.audio_name
    stream1.write('musicxml.pdf', pdf_path)
    file_path = pdf_path + '.pdf'
    new_saving_path = pdf_path + f" ({data}).pdf"
    os.rename(file_path, new_saving_path)
    os.rename(pdf_path + '.musicxml', new_saving_path[:-4] + '.musicxml')
    pdf_path = new_saving_path[:-4]

    audio_instance = audio_file
    notes_durations = notes_duration_idx
    algorithm_done = True


def play_song():
    global song_length
    pygame.mixer.music.load(path)
    pygame.mixer.music.play(loops=0)

    # call the play_time function to get song lenght
    play_time()

    # update slider to position
    slider_position = int(song_length)
    my_slider.config(to=slider_position, value=0)


def pause_song(is_paused):
    global paused
    paused = is_paused
    if paused:
        pygame.mixer.music.unpause()
        paused = False
    else:
        pygame.mixer.music.pause()
        paused = True


def stop_song():
    pygame.mixer.music.stop()

    # clear status_bar
    status_bar.config(text='')


def slide(x):
    global song_length


def open_musescore():
    global stream
    stream.show()


def open_pdf():
    global pdf_path
    file = pdf_path + '.pdf'
    subprocess.Popen([file], shell=True)


def create_midi():
    global stream, audio_instance, creation_time
    mf = music21.midi.translate.streamToMidiFile(stream)
    mf.open(os.getcwd() + '\Results\MIDI\\' + audio_instance.audio_name + creation_time + ".mid", 'wb')
    mf.write()
    mf.close()


def play_time():
    global path, audio_instance

    # grab current song elapsed time
    current_time = pygame.mixer.music.get_pos() / 1000

    # convert to time format
    converted_current_time = time.strftime('%M:%S', time.gmtime(current_time))

    # get song length
    global song_length
    song_length = librosa.get_duration(y=audio_instance.audio_data, sr=44100)

    # convert length to time format
    converted_song_length = time.strftime('%M:%S', time.gmtime(song_length))

    # output time to status bar
    status_bar.config(text=f"Minęło: {converted_current_time} z {converted_song_length} ")

    # update slider position value to current song position
    my_slider.config(value=int(current_time))
    # update time
    status_bar.after(1000, play_time)


def show_wave_plot(audio_path):
    global song_length
    audio_data, sr = librosa.load(audio_path, sr=44100)
    figure1 = plt.Figure(figsize=(10, 5), dpi=50)
    ax1 = figure1.add_subplot(111)
    bar1 = FigureCanvasTkAgg(figure1, mid_page_3)
    # bar1.get_tk_widget().place(x=25, y=80, width=924, height=250)
    bar1.get_tk_widget().place(x=20, y=80, width=924, height=250)

    librosa.display.waveshow(audio_data, alpha=0.5, sr=44100, ax=ax1)
    ax1.set_xlabel("Czas [s]")
    # ax1.get_yaxis().set_visible(False)
    ax1.set_ylabel("Amplituda")
    # print(int(song_length))
    ax1.set_xlim([0, len(audio_data) / 44100])
    figure1.tight_layout()


def show_detailed_plots():
    global audio_instance
    global notes_durations

    figure1 = plt.Figure(dpi=100, figsize=(10, 8))
    ax1 = figure1.add_subplot(221)
    ax2 = figure1.add_subplot(222)
    ax3 = figure1.add_subplot(223)
    ax4 = figure1.add_subplot(224)

    bar1 = FigureCanvasTkAgg(figure1, mid_page_4)
    bar1.get_tk_widget().place(x=25, y=10, width=974, height=638)
    librosa.display.waveshow(audio_instance.audio_data_filtered, alpha=0.5, sr=44100, ax=ax1)

    ax1.set_title(f'Przefiltrowany sygnał')
    ax1.set_xlabel("Czas [s]")
    ax1.set_ylabel("Amplituda")

    x = np.arange(0, audio_instance.audio_data.shape[0]) / audio_instance.sample_rate
    zeros = np.zeros(audio_instance.audio_data.shape[0])

    for sound in notes_durations:
        first_idx, second_idx = sound
        zeros[first_idx:second_idx] = 1
    collection = collections.BrokenBarHCollection.span_where(
        x, ymin=0, ymax=np.abs(audio_instance.audio_data).max(),
        where=zeros > 0, facecolor='orange',
        label='Obszar występowania\ndźwięku')
    ax2.add_collection(collection)
    librosa.display.waveshow(audio_instance.audio_data, sr=audio_instance.sample_rate, ax=ax2)
    ax2.set_xlabel('Czas [s]')
    ax2.set_ylabel('Amplituda')
    ax2.set_title(f'Przedziały występowania dźwięków')
    ax2.legend(loc='lower right')

    stft_data = np.abs(librosa.stft(audio_instance.audio_data))
    stft_data_db = librosa.amplitude_to_db(stft_data)
    img = librosa.display.specshow(stft_data_db, x_axis='time', y_axis='log', ax=ax3, sr=audio_instance.sample_rate)
    ax3.set_title(f"Spectrogram")
    ax3.set_xlabel("Czas [s]")
    ax3.set_ylabel("Częstotliwość [Hz]")
    figure1.colorbar(img, ax=ax3, format="%+2.0f dB")

    onset_frames = librosa.samples_to_frames(audio_instance.onsets, hop_length=audio_instance.hop_size)
    times = librosa.times_like(audio_instance.envelope, sr=audio_instance.sample_rate)

    ax4.plot(times, audio_instance.envelope, label='strumień widmowy')
    ax4.vlines(times[onset_frames], 0, audio_instance.envelope.max(), color='r', alpha=0.5, linestyle='--',
               label='wykryte nuty')
    ax4.legend(loc='upper right')
    ax4.set_ylabel("Amplituda")
    ax4.set_xlabel("Czas [s]")
    ax4.set_title(f"Wykryte miejsca kolejnych nut")
    figure1.tight_layout()


def handle_selection(event):
    print(event.widget.get())


if __name__ == "__main__":
    root = tk.Tk()  # Creating instance of Tk class
    root.title("Aplikacja muzyczna do przetwarzania ścieżek dźwiękowych w nuty")

    window_height = 768
    window_width = 1024
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    x = (screen_width / 2) - (window_width / 2)
    y = (screen_height / 2) - (window_height / 2)
    root.geometry(f"{window_width}x{window_height}+{int(x)}+{int(y - 30)}")

    label_font = ("Helvetica", "20")
    label_font_2 = ("Helvetica", "16")
    img_AGH = ImageTk.PhotoImage(Image.open('GUI/logoAGH.png').resize((40, 40)))
    img_sax = ImageTk.PhotoImage(Image.open('GUI/saxophone.png').resize((40, 40)))
    file_img = ImageTk.PhotoImage(Image.open('GUI/folder.png').resize((40, 40)))
    microphone_img = ImageTk.PhotoImage(Image.open('GUI/microphone.png').resize((40, 40)))
    quit_img = ImageTk.PhotoImage(Image.open('GUI/exit.png').resize((40, 40)))
    next_img = ImageTk.PhotoImage(Image.open('GUI/next.png').resize((40, 40)))
    cancel_img = ImageTk.PhotoImage(Image.open('GUI/back.png').resize((40, 40)))
    save_img = ImageTk.PhotoImage(Image.open('GUI/save.png').resize((40, 40)))
    graph_img = ImageTk.PhotoImage(Image.open('GUI/graph.png').resize((40, 40)))
    play_img = ImageTk.PhotoImage(Image.open('GUI/play.png').resize((55, 55)))
    pause_img = ImageTk.PhotoImage(Image.open('GUI/pause.png').resize((55, 55)))
    stop_img = ImageTk.PhotoImage(Image.open('GUI/stop.png').resize((55, 55)))
    stop_recording_img = ImageTk.PhotoImage(Image.open('GUI/stop_recording.png').resize((40, 40)))


    class TopFrame:
        def __init__(self, window):
            self.window = window
            self.top_frame = tk.Frame(self.window, width=1024, height=60, relief="groove", borderwidth=2)
            self.head_label_left = tk.Label(self.top_frame, text="MusicApp", image=img_AGH, compound="left",
                                            font=label_font, width=512, anchor="w")
            self.head_label_right = tk.Label(self.top_frame, image=img_sax, width=512, anchor="e")
            self.head_label_left.pack(side="left", fill="y")
            self.head_label_right.pack(side="left", fill="y")

            self.top_frame.pack(fill="x", expand=True)


    class MidFrame:
        def __init__(self, window):
            self.window = window
            self.mid_frame = tk.Frame(self.window, width=1024, height=658, background='#FFFFFF')


    class BottomFrame:
        def __init__(self, window):
            self.window = window
            self.bot_frame = tk.Frame(self.window, width=1024, height=60, relief="groove", borderwidth=2)


    class Page:
        def __init__(self, top_frame, mid_frame, bot_frame):
            self.top_frame = top_frame
            self.mid_frame = mid_frame
            self.bot_frame = bot_frame


    top_frame = tk.Frame(root, width=1024, height=50, relief="groove", borderwidth=2, background="#F0F0F0")

    head_label_left = tk.Label(top_frame, text="Aplikacja muzyczna", image=img_AGH, compound="left", font=label_font,
                               width=512,
                               anchor="w", background="#F0F0F0")
    head_label_right = tk.Label(top_frame, text=" ", compound="left", image=img_sax, width=512, anchor="e",
                                background="#F0F0F0")

    head_label_left.place(x=0, y=0, width=512, height=50)
    head_label_right.place(x=512, y=0, width=512, height=50)

    top_frame.pack(fill="x")
    # ----------------------------------------------------------------------------------------------------------------
    mid_page_1 = tk.Frame(root, width=1024, height=658, background='#FFFFFF')
    main_text = tk.Label(mid_page_1, text="Witaj w aplikacji muzycznej do przetwarzania\n"
                                          "ścieżek dźwiękowych w nuty!\n\n"
                                          "Aby przejść dalej wybierz lub nagraj plik audio.",
                         font=label_font, background='#FFFFFF')
    load_audio_button = tk.Button(mid_page_1, text=" Wybierz plik", font=label_font, image=file_img,
                                  compound="left", command=lambda: [select_file(), check_path()], cursor="hand2")
    record_audio_button = tk.Button(mid_page_1, text="Nagraj ścieżkę audio", font=label_font,
                                    image=microphone_img, compound="left", cursor="hand2",
                                    command=lambda: [switch_to_record_frame(),
                                                     ])
    main_text.place(x=112, y=100, width=800, height=300)
    load_audio_button.place(x=262, y=420, width=500, height=60, )
    record_audio_button.place(x=262, y=530, width=500, height=60, )
    mid_page_1.pack(fill="x")
    # ----------------------------------------------------------------------------------------------------------------
    mid_page_12 = tk.Frame(root, width=1024, height=658, background='#FFFFFF')
    recording_frame = tk.Frame(mid_page_12, borderwidth=4, background='#FFFFFF', relief="groove")
    recording_frame.place(x=237, y=39, width=550, height=580)
    record_audio_label = tk.Label(recording_frame, text="Kliknij poniższy przycisk aby rozpocząć nagrywanie:",
                                  font=label_font_2, background='#FFFFFF', anchor="w")

    record_audio_button_2 = tk.Button(recording_frame, text=" Rozpocznij nagrywanie", font=label_font_2,
                                      image=microphone_img, compound="left",
                                      command=lambda: [handle_recording(), switch_button_state(record_audio_button_2)],
                                      cursor="hand2")
    time_label = tk.Label(recording_frame, text="00:00:00", background='#FFFFFF', font=label_font_2)

    stop_recording_label = tk.Label(recording_frame, text="Kliknij poniższy przycisk aby zatrzymać nagrywanie:",
                                    font=label_font_2, background='#FFFFFF', anchor="w")

    stop_recording_button = tk.Button(recording_frame, text=" Zatrzymaj nagrywanie", state='disabled',
                                      font=label_font_2,
                                      image=stop_recording_img, compound="left",
                                      command=lambda: [handle_recording(), stop_recording(),
                                                       switch_button_state(stop_recording_button)],
                                      cursor="hand2")

    name_audio_label = tk.Label(recording_frame, text="Wprowadź nazwę: ",
                                font=label_font_2, background='#FFFFFF', anchor="w")

    recorded_name = tk.StringVar()
    file_name_entry = tk.Entry(recording_frame, font=label_font_2, textvariable=recorded_name, bd=5, state='disabled')

    save_recorded_audio = tk.Button(recording_frame, text="Zapisz plik", font=label_font_2, compound="left",
                                    cursor="hand2", state='disabled', command=lambda: [save_recording()])
    warning = tk.Label(recording_frame, text="",
                       font=label_font_2, background='#FFFFFF', fg="red")

    close_recording_frame = tk.Button(recording_frame, text="Zamknij okno", font=label_font_2, compound="left",
                                      cursor="hand2",
                                      command=lambda: [switch_to_record_frame(), clear_recording_window(),
                                                       check_path()])
    repeat_recording_button = tk.Button(recording_frame, text="Powtórz nagrywnie", font=label_font_2, compound="left",
                                        cursor="hand2", command=lambda: [clear_recording_window()])

    record_audio_label.place(x=25, y=25, width=500, height=25)
    record_audio_button_2.place(x=135, y=75, width=280, height=50)
    time_label.place(x=135, y=140, width=280, height=25)
    stop_recording_label.place(x=25, y=200, width=500, height=25)
    stop_recording_button.place(x=135, y=250, width=280, height=50)
    name_audio_label.place(x=25, y=325, width=250, height=50)
    file_name_entry.place(x=225, y=325, width=250, height=50)
    warning.place(x=25, y=390, width=500, height=25)
    save_recorded_audio.place(x=135, y=430, width=280, height=50)
    repeat_recording_button.place(x=25, y=505, width=225, height=50)
    close_recording_frame.place(x=290, y=505, width=225, height=50)

    # ----------------------------------------------------------------------------------------------------------------
    bottom_page_1 = tk.Frame(root, width=1024, height=60, relief="groove", borderwidth=2)
    quit_button = tk.Button(bottom_page_1, text="Wyjście", image=quit_img, compound="left", font=label_font,
                            width=512, command=root.destroy, cursor="hand2")  # anchor="w"
    next_button_page_1 = tk.Button(bottom_page_1, text="Dalej", image=next_img, compound="right", font=label_font,
                                   width=512, state="disabled", command=move_next_page, cursor="hand2")  # anchor="e"

    quit_button.place(x=0, y=0, width=512, height=60)
    next_button_page_1.place(x=512, y=0, width=512, height=60)

    bottom_page_1.pack(fill="x", expand=True)
    # ----------------------------------------------------------------------------------------------------------------

    mid_page_2 = tk.Frame(root, width=1024, height=658, background='#FFFFFF')
    main_text = tk.Label(mid_page_2, text="Ustaw wartości poniższych parametrów",
                         font=label_font, background="#FFFFFF")
    save_setting_button = tk.Button(mid_page_2, text=" Uruchom działanie aplikacji", font=label_font, image=save_img,
                                    compound="left", cursor="hand2",
                                    command=lambda: [run_application(path, audio_name),
                                                     show_wave_plot(path),
                                                     show_detailed_plots(),
                                                     switch_button_state(next_button_page_2)])
    settings_frame = tk.Frame(mid_page_2, borderwidth=4, background='#FFFFFF', relief="groove")

    shortest_note_label = tk.Label(mid_page_2, text="shortest_note:", font=label_font_2,
                                   background='#FFFFFF').place(x=282, y=190, height=30)
    threshold_label_label = tk.Label(mid_page_2, text="top_db:", font=label_font_2,
                                     background='#FFFFFF').place(x=282, y=240, height=30)
    windows_size = tk.Label(mid_page_2, text="window_size:", font=label_font_2,
                            background='#FFFFFF').place(x=282, y=290, height=30)
    hop_size_note_label = tk.Label(mid_page_2, text="hop_size:", font=label_font_2,
                                   background='#FFFFFF').place(x=282, y=340, height=30)
    min_frequency_label = tk.Label(mid_page_2, text="min_freq:", font=label_font_2,
                                   background='#FFFFFF').place(x=282, y=390)
    max_frequency_label = tk.Label(mid_page_2, text="max_freq:", font=label_font_2,
                                   background='#FFFFFF').place(x=282, y=440, height=30)

    selected_shortest_note = tk.StringVar()
    selected_threshold = tk.StringVar()
    selected_window_size = tk.StringVar()
    selected_hop_size = tk.StringVar()
    selected_min_frequency = tk.StringVar()
    selected_max_frequency = tk.StringVar()

    shortest_note_combobox = ttk.Combobox(mid_page_2, textvariable=selected_shortest_note,
                                          values=["1/8", "1/16", "1/32", "1/64"],
                                          font=label_font_2, state='readonly')
    threshold_combobox = ttk.Combobox(mid_page_2, textvariable=selected_threshold,
                                      values=["10", "20", "30", "40", "50", "60"], font=label_font_2,
                                      state='readonly')
    windows_size_combobox = ttk.Combobox(mid_page_2, textvariable=selected_window_size, state='readonly',
                                         values=["512", "1024", "2048", "4096", "8192"], font=label_font_2)
    hop_size_note_combobox = ttk.Combobox(mid_page_2, textvariable=selected_hop_size,
                                          values=["256", "512", "1024", "2048", "4096"],
                                          font=label_font_2, state='readonly')
    min_frequency_combobox = ttk.Combobox(mid_page_2, textvariable=selected_min_frequency,
                                          values=["50", "100", "200", "500"],
                                          font=label_font_2, state='readonly')
    max_frequency_combobox = ttk.Combobox(mid_page_2, textvariable=selected_max_frequency,
                                          values=["500", "1000", "1500", "2000", "2500", "3000"],
                                          state='readonly', font=label_font_2)

    shortest_note_combobox.place(x=482, y=190, width=260, height=30)
    threshold_combobox.place(x=482, y=240, width=260, height=30)
    windows_size_combobox.place(x=482, y=290, width=260, height=30)
    hop_size_note_combobox.place(x=482, y=340, width=260, height=30)
    min_frequency_combobox.place(x=482, y=390, width=260, height=30)
    max_frequency_combobox.place(x=482, y=440, width=260, height=30)

    shortest_note_combobox.current(0)
    threshold_combobox.current(2)
    windows_size_combobox.current(1)
    hop_size_note_combobox.current(1)
    min_frequency_combobox.current(0)
    max_frequency_combobox.current(4)

    main_text.place(x=262, y=100, width=500, height=50)
    settings_frame.place(x=262, y=170, width=500, height=320)
    save_setting_button.place(x=262, y=530, width=500, height=60)

    shortest_note_combobox.bind("<<ComboboxSelected>>", handle_selection)
    threshold_combobox.bind("<<ComboboxSelected>>", handle_selection)
    windows_size_combobox.bind("<<ComboboxSelected>>", handle_selection)
    hop_size_note_combobox.bind("<<ComboboxSelected>>", handle_selection)
    min_frequency_combobox.bind("<<ComboboxSelected>>", handle_selection)
    max_frequency_combobox.bind("<<ComboboxSelected>>", handle_selection)
    # mid_frame_2.pack(fill="x")
    # ----------------------------------------------------------------------------------------------------------------
    bottom_page_2 = tk.Frame(root, width=1024, height=60, relief="groove", borderwidth=2)

    cancel_button_page_2 = tk.Button(bottom_page_2, text="Powrót", image=cancel_img, compound="left", font=label_font,
                                     width=512, cursor="hand2", command=move_previous_page)  # anchor="w"
    next_button_page_2 = tk.Button(bottom_page_2, text="Dalej", image=next_img, compound="right", font=label_font,
                                   width=512, state="disabled", cursor="hand2",
                                   command=lambda: [move_next_page()])  # anchor="e"

    cancel_button_page_2.place(x=0, y=0, width=512, height=60)
    next_button_page_2.place(x=512, y=0, width=512, height=60)
    # bottom_frame_2.pack(fill="x", expand=True)
    # ----------------------------------------------------------------------------------------------------------------
    mid_page_3 = tk.Frame(root, width=1024, height=658, background='#FFFFFF')
    audio_name = tk.StringVar()
    sound_track_label = tk.Label(mid_page_3, text="Ścieżka dźwiękowa:", font=label_font_2, anchor="w",
                                 background='#FFFFFF').place(y=25, x=50, height=30, width=200)
    audio_name_label = tk.Label(mid_page_3, textvariable=audio_name, font=label_font_2, anchor="w",
                                background='#FFFFFF').place(y=25, x=250, height=30, width=500)

    pygame.mixer.init()

    mp3_frame = tk.Frame(mid_page_3, borderwidth=4, background='#FFFFFF', relief="groove")
    # mp3_frame.place(x=344, y=395, width=336, height=70)

    play_button = tk.Button(mid_page_3, image=play_img, borderwidth=0, background="#FFFFFF",
                            activebackground="#FFFFFF", cursor="hand2", command=play_song).place(x=362, y=400, width=60,
                                                                                                 height=60)
    pause_button = tk.Button(mid_page_3, image=pause_img, borderwidth=0, background="#FFFFFF",
                             activebackground="#FFFFFF", cursor="hand2", command=lambda: [pause_song(paused)]).place(
        x=482, y=400,
        width=60,
        height=60)
    stop_button = tk.Button(mid_page_3, image=stop_img, borderwidth=0, background="#FFFFFF",
                            activebackground="#FFFFFF", cursor="hand2", command=stop_song).place(x=602, y=400, width=60,
                                                                                                 height=60)

    status_bar = tk.Label(mid_page_3, text='', relief="groove", background="#FFFFFF", borderwidth=0)
    status_bar.place(y=50, x=750, width=224, height=30)

    style = ttk.Style()
    style.configure("TScale", background="#FFFFFF")
    my_slider = ttk.Scale(mid_page_3, from_=0, to=100, orient=tk.HORIZONTAL, value=0, command=slide, length=840,
                          style="TScale")
    my_slider.place(x=90, y=350, height=30)

    show_pdf = tk.Button(mid_page_3, text="Pokaż nuty (pdf)", font=label_font, compound="left", cursor="hand2",
                         command=lambda: [open_pdf()])
    show_musescore = tk.Button(mid_page_3, text="Pokaż nuty (Musescore)", font=label_font, compound="left",
                               cursor="hand2", command=lambda: [open_musescore()])
    generate_midi = tk.Button(mid_page_3, text="Generuj midi", font=label_font, cursor="hand2", command=create_midi)

    show_pdf.place(x=90, y=486, width=400, height=60)
    show_musescore.place(x=534, y=486, width=400, height=60)
    generate_midi.place(x=312, y=572, width=400, height=60)
    # mid_page_3.pack(fill="x")
    # ----------------------------------------------------------------------------------------------------------------
    bottom_page_3 = tk.Frame(root, width=1024, height=60, relief="groove", borderwidth=2)
    cancel_button_page_3 = tk.Button(bottom_page_3, text="Powrót", image=cancel_img, compound="left", font=label_font,
                                     width=512, cursor="hand2", command=move_previous_page)
    show_plots_button_page_3 = tk.Button(bottom_page_3, text="Pokaż wykresy", image=next_img, compound="right",
                                         cursor="hand2", font=label_font, width=512, command=move_next_page)

    cancel_button_page_3.place(x=0, y=0, width=512, height=60)
    show_plots_button_page_3.place(x=512, y=0, width=512, height=60)

    # bottom_page_3.pack(fill="x", expand=True)
    # ----------------------------------------------------------------------------------------------------------------
    mid_page_4 = tk.Frame(root, width=1024, height=658, background='#FFFFFF')
    # mid_page_4.pack(fill="x")
    # ----------------------------------------------------------------------------------------------------------------

    bottom_page_4 = tk.Frame(root, width=1024, height=60, relief="groove", borderwidth=2)

    cancel_button_page_4 = tk.Button(bottom_page_4, text="Powrót", image=cancel_img, compound="left", font=label_font,
                                     width=1024, cursor="hand2", command=move_previous_page)

    cancel_button_page_4.place(x=0, y=0, width=1024, height=60)
    root.mainloop()
