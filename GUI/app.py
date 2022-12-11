import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import librosa
import pygame

count_pages = 0
path = None
audio_name = None


def move_next_page():
    global count_pages
    count_pages += 1
    if count_pages == 1:
        mid_page_2.pack(fill="x")
        bottom_page_2.pack(fill="x", expand=True)
        mid_page_1.forget()
        bottom_page_1.forget()
    if count_pages == 2:
        mid_page_3.pack(fill="x")
        bottom_page_3.pack(fill="x", expand=True)
        mid_page_2.forget()
        bottom_page_2.forget()


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


def select_file():
    global path, audio_name
    filetypes = (
        ("Audio files", ".wav")
    )
    path = fd.askopenfilename(
        title='Open a file',
        initialdir='D:\Studia\MusicApp\Audio')

    audio_name.set(path.split('/')[-1])
    # filetypes=filetypes)
    print(path)
    print(audio_name)
    # showinfo(
    #     title='Selected File',
    #     message=filename
    # )


def switch_button_state(button_type):
    if button_type["state"] == "normal":
        if button_type is next_button_page_1 or button_type is next_button_page_2:
            pass
        else:
            button_type["state"] = "disabled"
    else:
        button_type["state"] = "normal"


def show_wave_plot(audio_path, audio_name):
    audio_data, sr = librosa.load(audio_path)
    figure1 = plt.Figure(figsize=(10, 5), dpi=50)
    ax1 = figure1.add_subplot(111)
    bar1 = FigureCanvasTkAgg(figure1, mid_page_3)
    bar1.get_tk_widget().place(x=50, y=100, width=924, height=200)
    librosa.display.waveshow(audio_data, alpha=0.5, sr=44100, ax=ax1)
    ax1.set_title(f'Wave plot of {audio_name.get()}')
    ax1.set_xlabel("Czas [s]")
    ax1.set_ylabel("Amplituda")


root = tk.Tk()  # Creating instance of Tk class
root.title("MusicApp")

window_height = 768
window_width = 1024
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

x = (screen_width / 2) - (window_width / 2)
y = (screen_height / 2) - (window_height / 2)
root.geometry(f"{window_width}x{window_height}+{int(x)}+{int(y - 30)}")

label_font = ("Helvetica", "20")
label_font_2 = ("Helvetica", "16")
img_AGH = ImageTk.PhotoImage(Image.open('images/Znak_graficzny_AGH.svg.png').resize((40, 40)))
img_sax = ImageTk.PhotoImage(Image.open('images/saxophone.png').resize((40, 40)))
file_img = ImageTk.PhotoImage(Image.open('images/folder.png').resize((40, 40)))
microphone_img = ImageTk.PhotoImage(Image.open('images/microphone.png').resize((40, 40)))
quit_img = ImageTk.PhotoImage(Image.open('images/exit.png').resize((40, 40)))
next_img = ImageTk.PhotoImage(Image.open('images/next.png').resize((40, 40)))
cancel_img = ImageTk.PhotoImage(Image.open('images/back.png').resize((40, 40)))
save_img = ImageTk.PhotoImage(Image.open('images/save.png').resize((40, 40)))
graph_img = ImageTk.PhotoImage(Image.open('images/graph.png').resize((40, 40)))
play_img = ImageTk.PhotoImage(Image.open('images/play.png').resize((40, 40)))
pause_img = ImageTk.PhotoImage(Image.open('images/pause.png').resize((40, 40)))
stop_img = ImageTk.PhotoImage(Image.open('images/stop.png').resize((40, 40)))


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
    def __init__(self, window, top_frame, mid_frame, bot_frame):
        self.top_frame = top_frame
        self.mid_frame = mid_frame
        self.bot_frame = bot_frame


# ----------------------------------------------------------------------------------------------------------------
top_frame = tk.Frame(root, width=1024, height=60, relief="groove", borderwidth=2)

head_label_left = tk.Label(top_frame, text="MusicApp", image=img_AGH, compound="left", font=label_font, width=512,
                           anchor="w")
head_label_right = tk.Label(top_frame, image=img_sax, width=512, anchor="e")
head_label_left.pack(side="left", fill="y")
head_label_right.pack(side="left", fill="y")

top_frame.pack(fill="x", expand=True)
# ----------------------------------------------------------------------------------------------------------------

mid_page_1 = tk.Frame(root, width=1024, height=658, background='#FFFFFF')
main_text = tk.Label(mid_page_1, text="Witaj w aplikacji muzycznej do przetwarzania\n"
                                      "ścieżek dźwiękowych w nuty!\n\n"
                                      "Aby przejść dalej wybierz lub nagraj plik audio.",
                     font=label_font, background='#FFFFFF')
load_audio_button = tk.Button(mid_page_1, text=" Wybierz plik", font=label_font, image=file_img,
                              compound="left", command=lambda: [select_file(),
                                                                switch_button_state(next_button_page_1)])
record_audio_button = tk.Button(mid_page_1, text="Nagraj ścieżkę audio", font=label_font,
                                image=microphone_img, compound="left")
main_text.place(x=112, y=100, width=800, height=300)
load_audio_button.place(x=262, y=420, width=500, height=60, )
record_audio_button.place(x=262, y=530, width=500, height=60, )
mid_page_1.pack(fill="x")

# ----------------------------------------------------------------------------------------------------------------
bottom_page_1 = tk.Frame(root, width=1024, height=60, relief="groove", borderwidth=2)
quit_button = tk.Button(bottom_page_1, text="Wyjście", image=quit_img, compound="left", font=label_font,
                        width=512, command=root.destroy)  # anchor="w"
next_button_page_1 = tk.Button(bottom_page_1, text="Dalej", image=next_img, compound="right", font=label_font,
                               width=512, state="disabled", command=move_next_page)  # anchor="e"

quit_button.grid(row=0, column=0)
next_button_page_1.grid(row=0, column=1)

bottom_page_1.pack(fill="x", expand=True)
# ----------------------------------------------------------------------------------------------------------------

mid_page_2 = tk.Frame(root, width=1024, height=658, background='#FFFFFF')
main_text = tk.Label(mid_page_2, text="Ustaw wartości poniższych parametrów",
                     font=label_font, background="#FFFFFF")
save_setting_button = tk.Button(mid_page_2, text=" Uruchom działanie aplikacji", font=label_font, image=save_img,
                                compound="left", command=lambda: [switch_button_state(next_button_page_2)])
settings_frame = tk.Frame(mid_page_2, borderwidth=2, background='#FFFFFF', relief="groove")

shortest_note_label = tk.Label(mid_page_2, text="shortest_note:", font=label_font_2,
                               background='#FFFFFF').place(x=282, y=190, height=30)
threshold_label_label = tk.Label(mid_page_2, text="threshold:", font=label_font_2,
                                 background='#FFFFFF').place(x=282, y=240, height=30)
windows_size = tk.Label(mid_page_2, text="windows_size:", font=label_font_2,
                        background='#FFFFFF').place(x=282, y=290, height=30)
hop_size_note_label = tk.Label(mid_page_2, text="hop_size:", font=label_font_2,
                               background='#FFFFFF').place(x=282, y=340, height=30)
min_frequency_label = tk.Label(mid_page_2, text="min_frequency:", font=label_font_2,
                               background='#FFFFFF').place(x=282, y=390)
max_frequency_label = tk.Label(mid_page_2, text="max_frequency:", font=label_font_2,
                               background='#FFFFFF').place(x=282, y=440, height=30)

selected_shortest_note = tk.StringVar()
selected_threshold = tk.StringVar()
selected_window_size = tk.StringVar()
selected_hop_size = tk.StringVar()
selected_min_frequency = tk.StringVar()
selected_max_frequency = tk.StringVar()

shortest_note_combobox = ttk.Combobox(mid_page_2,
                                      values=["ósemka", "szesnastka", "trzydziestodwójka", "sześćdziesięcioczwórka"],
                                      textvariable=selected_shortest_note,
                                      font=label_font_2, state='readonly')
threshold_combobox = ttk.Combobox(mid_page_2, values=["10", "20", "30", "40", "50", "60"], font=label_font_2,
                                  state='readonly')
windows_size_combobox = ttk.Combobox(mid_page_2, state='readonly',
                                     values=["512", "1024", "2048", "4096", "8192"], font=label_font_2)
hop_size_note_combobox = ttk.Combobox(mid_page_2, values=["256", "512", "1024", "2048", "4096"],
                                      font=label_font_2, state='readonly')
min_frequency_combobox = ttk.Combobox(mid_page_2, values=["50 Hz", "100 Hz", "200 Hz", "500 Hz"],
                                      font=label_font_2, state='readonly')
max_frequency_combobox = ttk.Combobox(mid_page_2,
                                      values=["500 Hz", "1000 Hz", "1500 Hz", "2000 Hz", "2500 Hz", "3000 Hz"],
                                      state='readonly', font=label_font_2)

shortest_note_combobox.place(x=482, y=190, width=260, height=30)
threshold_combobox.place(x=482, y=240, width=260, height=30)
windows_size_combobox.place(x=482, y=290, width=260, height=30)
hop_size_note_combobox.place(x=482, y=340, width=260, height=30)
min_frequency_combobox.place(x=482, y=390, width=260, height=30)
max_frequency_combobox.place(x=482, y=440, width=260, height=30)

shortest_note_combobox.current(0)
threshold_combobox.current(2)
windows_size_combobox.current(2)
hop_size_note_combobox.current(2)
min_frequency_combobox.current(0)
max_frequency_combobox.current(4)

main_text.place(x=262, y=100, width=500, height=50)
settings_frame.place(x=262, y=170, width=500, height=320)
save_setting_button.place(x=262, y=530, width=500, height=60)


def handle_selection(event):
    print(event.widget.get())


shortest_note_combobox.bind("<<ComboboxSelected>>", handle_selection)
threshold_combobox.bind("<<ComboboxSelected>>", handle_selection)
windows_size_combobox.bind("<<ComboboxSelected>>", handle_selection)
hop_size_note_combobox.bind("<<ComboboxSelected>>", handle_selection)
min_frequency_combobox.bind("<<ComboboxSelected>>", handle_selection)
max_frequency_combobox.bind("<<ComboboxSelected>>", handle_selection)
# mid_frame_1.pack(fill="x")
# ----------------------------------------------------------------------------------------------------------------
bottom_page_2 = tk.Frame(root, width=1024, height=60, relief="groove", borderwidth=2)

cancel_button_page_2 = tk.Button(bottom_page_2, text="Powrót", image=cancel_img, compound="left", font=label_font,
                                 width=512, command=move_previous_page)  # anchor="w"
next_button_page_2 = tk.Button(bottom_page_2, text="Dalej", image=next_img, compound="right", font=label_font,
                               width=512, state="disabled",
                               command=lambda: [show_wave_plot(path, audio_name), move_next_page()])  # anchor="e"

cancel_button_page_2.grid(row=0, column=0)
next_button_page_2.grid(row=0, column=1)
# bottom_frame_1.pack(fill="x", expand=True)
# ----------------------------------------------------------------------------------------------------------------
mid_page_3 = tk.Frame(root, width=1024, height=658, background='#FFFFFF')
# main_text = tk.Label(mid_page_3, text="", font=label_font, background='#FFFFFF')
audio_name = tk.StringVar()
audio_name_label = tk.Label(mid_page_3, textvariable=audio_name, font=label_font_2, anchor="w").place(y=50,
                                                                                                      x=50,
                                                                                                      height=30,
                                                                                                      width=924)

sheetmusic_button = tk.Button(mid_page_3, text="Generuj nuty", font=label_font,
                              compound="left")
show_pdf = tk.Button(mid_page_3, text="Pokaż nuty (pdf)", font=label_font, compound="left")
show_musescore = tk.Button(mid_page_3, text="Pokaż nuty (Musescore)", font=label_font, compound="left")

sheetmusic_button.place(x=262, y=420, width=500, height=60, )
show_pdf.place(x=50, y=530, width=437, height=60)
show_musescore.place(x=537, y=530, width=437, height=60)

# mid_page_3.pack(fill="x")
# ----------------------------------------------------------------------------------------------------------------
bottom_page_3 = tk.Frame(root, width=1024, height=60, relief="groove", borderwidth=2)
cancel_button_page_3 = tk.Button(bottom_page_3, text="Powrót", image=cancel_img, compound="left", font=label_font,
                                 width=512, command=move_previous_page)
show_plots_button_page_3 = tk.Button(bottom_page_3, text="Pokaż wykresy", image=graph_img, compound="right",
                                     font=label_font,
                                     width=512)

cancel_button_page_3.grid(row=0, column=0)
show_plots_button_page_3.grid(row=0, column=1)

# bottom_page_3.pack(fill="x", expand=True)
root.mainloop()
