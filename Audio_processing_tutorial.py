import numpy as np
import librosa
import librosa.display
from glob import glob
import matplotlib.pyplot as plt

def amplitude_envelope(signal, frame_size, hop_length):
    """Fancier Python code to calculate the amplitude envelope of a signal with a given frame size."""
    return np.array([max(signal[i:i + frame_size]) for i in range(0, len(signal), hop_length)])


if __name__ == "__main__":
    audio_files = np.sort(glob('Audio/*'))
    print(audio_files)
    audio_path = audio_files[4]
    audio_data, sr = librosa.load(audio_path)  # y - raw data of audio file, sr - sample rate of audio file
    audio_name = audio_path[6:]

    FRAME_SIZE = 1024
    HOP_LENGTH = 512
    ae_audio_data = amplitude_envelope(audio_data, FRAME_SIZE, HOP_LENGTH)
    print(len(ae_audio_data))
    frames = range(len(ae_audio_data))
    t = librosa.frames_to_time(frames, hop_length=HOP_LENGTH)

    plt.figure(figsize=(10, 5))
    librosa.display.waveshow(audio_data, alpha=0.5)
    plt.plot(t, ae_audio_data, color="r")
    plt.ylim((-1, 1))
    plt.title(f"{audio_name} with amplitude envelope")
    plt.show()