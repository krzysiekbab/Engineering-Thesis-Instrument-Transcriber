import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import IPython.display as ipd
from playsound import playsound
import os
from constants import *
from functions import *
# from audio_class import Audio
import music21
import matplotlib.collections as collections
from matplotlib.widgets import Slider


# %%

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


# Pobranie adresu pliku
directory = os.getcwd()
AUDIO_NAME = 'Autumn leaves - saksofon.wav'
AUDIO_PATH = directory + AUDIO_NAME

# Ustawienie wartości parametrów
WINDOW_SIZE = 1024
HOP_SIZE = 512
TOP_DB = 30
FREQUENCY_RANGE = (50, 2500)
SAMPLE_RATE = 44100
SHORTEST_NOTE = '1/8'

# Stworzenie instancji klasy
audio_file = Audio(audio_path=AUDIO_PATH, audio_name=AUDIO_NAME, sample_rate=SAMPLE_RATE, hop_size=HOP_SIZE,
                   window_size=WINDOW_SIZE, top_db=TOP_DB, frequency_range=FREQUENCY_RANGE,
                   shortest_note=SHORTEST_NOTE)
