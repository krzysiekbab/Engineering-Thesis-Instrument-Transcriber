import matplotlib.pyplot as plt  # to plot
import numpy as np
from scipy.fft import fft, fftfreq, rfft, rfftfreq
from scipy import signal
import wave
import scipy as sp
import os

import librosa
import librosa.display
import matplotlib.collections as collections
from constants import SAMPLE_RATE, FREQUENCY_RANGE


def generate_sine_wave(freq, sample_rate, duration, amplitude=1, phase=0):
    x = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    frequencies = x * freq
    # 2pi because np.sin takes radians
    y = amplitude * np.sin((2 * np.pi) * frequencies + phase)
    return x, y


def plot_quantization():
    """
    Quantization plot
    """
    f = 50  # Hz
    t = np.linspace(0, 0.1, 200)
    x1 = np.sin(2 * np.pi * f * t)
    s_rate = 500  # Hz

    T = 1 / s_rate  # 1/500 [s]
    n = np.arange(0, 0.1 / T)
    nT = n * T
    x2 = np.sin(2 * np.pi * f * nT)  # Since for sampling t = nT.

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))


    # ALT + J
    ax[0].plot(t, x1, 'g', label='Sygnał sinusoidalny o częstotliwości 50 [Hz]', )
    ax[0].text(-0.12, 0.5, s='a)', transform=ax[0].transAxes, va='top', ha='right')
    ax[0].legend(loc='upper right')
    ax[0].set_ylabel("Amplituda")
    ax[0].set_xlabel("Czas [s]")
    # ax[0].set_ylim(-1.5, 2)
    ax[0].grid()

    ax[1].plot(t, x1, 'g', label='Sygnał sinusoidalny o częstotliwości 50 [Hz]')
    ax[0].text(-0.12, 0.5, s='b)', transform=ax[1].transAxes, va='top', ha='right')
    ax[1].stem(nT, x2, 'o', label='Sygnał spróbkowany')
    ax[1].legend(loc='upper right')
    ax[1].set_xlabel("Czas [s]")
    ax[1].set_ylabel("Amplituda")
    ax[1].legend(loc='upper right')
    # ax[1].set_ylim(-1.5, 2)
    ax[1].grid()

    plt.show()


def plot_aliasing():
    """
    Shows aliasing problem when sampling frequency:
    f_s < 2f_m
    Nyquist frequency.
    :return:
    """

    # Oryginal signal: f = 50 [Hz]

    f = 1  # Hz
    t_max = 2
    t = np.linspace(0, t_max, 400)
    x1 = np.sin(2 * np.pi * f * t)
    s_rate = 10  # Hz

    f2 = 11  # Hz
    x2 = np.sin(2 * np.pi * f2 * t)

    T = 1 / s_rate  # 1/500 [s]
    n = np.arange(0, t_max / T)
    nT = n * T
    x3 = np.sin(2 * np.pi * f * nT)  # Since for sampling t = nT.

    plt.figure(figsize=(10, 4))
    plt.plot(t, x1, 'g', label=f'Sygnał sinusoidalny, $f_0$ = {f} [Hz]')
    plt.plot(t, x2, 'y', label=f'Sygnał sinusoidalny, $f$ = {f2} [Hz]')
    plt.stem(nT, x3, 'o', label=f'Spróbkowany sygnał $f_0$, $f_s$ = {s_rate} [Hz]')
    plt.legend(loc='upper right')
    # plt.legend(bbox_to_anchor=(1.04, 0.5), mode='expand', nrow=3, loc="center left")
    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda")
    plt.ylim(-1.5, 2)
    plt.grid()
    plt.show()


def dtf_leakage():
    SAMPLE_RATE = 44100
    DURATION = 1
    FREQUENCY = 3

    x, y = generate_sine_wave(FREQUENCY, SAMPLE_RATE, DURATION)

    N = SAMPLE_RATE * DURATION

    yf = rfft(y)
    xf = rfftfreq(N, 1 / SAMPLE_RATE)

    number_of_samples = 64
    T = 1 / number_of_samples  # 1/64 [s]
    n = np.arange(0, DURATION / T)  # wektor liczb całkowitych o długości T
    nT = n * T  # normalizacja do przedziału (0, 1)
    y2 = np.sin(2 * np.pi * FREQUENCY * nT)  # Since for sampling t = nT.

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    ax[0].plot(x, y, 'g', label='sygnał sinusoidalny, f = 3 [Hz]')
    ax[0].plot(nT, y2, 'bo', label='spróbkowany sygnał, N = 64')
    ax[0].text(-0.12, 0.5, s='a)', transform=ax[0].transAxes, va='top', ha='right')
    ax[0].legend(loc='upper right')
    ax[0].set_xlabel("Czas [s]")
    ax[0].set_ylabel("Amplituda")
    ax[0].set_ylim(-1.5, 2)
    ax[0].grid()

    ax[1].stem(xf[:32], np.abs(yf)[:32], 'k', basefmt="-k")
    ax[1].text(-0.12, 0.5, s='b)', transform=ax[1].transAxes, va='top', ha='right')
    ax[1].grid()
    ax[1].set_xlabel("Częstotliwość [Hz]")
    ax[1].set_ylabel("Moduł wartości DFT")
    plt.subplots_adjust(
        bottom=0.1,
        top=0.95)
    plt.show()

    # ---------------------------------------------------------------------------------

    DURATION = 1.2
    x, y = generate_sine_wave(FREQUENCY, SAMPLE_RATE, DURATION)
    print(len(x), len(y))
    N = int(SAMPLE_RATE * DURATION)

    yf = rfft(y)
    xf = rfftfreq(N, 1 / SAMPLE_RATE)

    number_of_samples = 64
    T = 1 / number_of_samples  # 1/64 [s]
    n = np.arange(0, DURATION / T)  # wektor liczb całkowitych o długości T
    nT = n * T  # normalizacja do przedziału (0, 1)
    y2 = np.sin(2 * np.pi * FREQUENCY * nT)  # Since for sampling t = nT.

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    ax[0].plot(x, y, 'g', label='sygnał sinusoidalny, f = 3 [Hz]')
    ax[0].plot(nT, y2, 'bo', label='spróbkowany sygnał, N = 64')
    ax[0].text(-0.12, 0.5, s='a)', transform=ax[0].transAxes, va='top', ha='right')
    ax[0].legend(loc='upper right')
    ax[0].set_xlabel("Czas [s]")
    ax[0].set_ylabel("Amplituda")
    ax[0].set_ylim(-1.5, 2)
    ax[0].grid()

    ax[1].stem(xf[:32], np.abs(yf)[:32], 'k', basefmt="-k")
    ax[1].text(-0.12, 0.5, s='b)', transform=ax[1].transAxes, va='top', ha='right')
    # ax[1].set_ylabel("Amplituda FFT |X(f)|")
    ax[1].set_xlabel("Częstotliwość [Hz]")
    ax[1].set_ylabel("Moduł wartości DFT")
    ax[1].grid()
    plt.subplots_adjust(
        bottom=0.1,
        top=0.95)

    plt.show()


def window_functions():
    SAMPLE_RATE = 100
    DURATION = 3
    FREQUENCY = 10

    x, y = generate_sine_wave(FREQUENCY, SAMPLE_RATE, DURATION)

    fig = plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid(shape=(4, 2), loc=(0, 0), colspan=2)
    ax2 = plt.subplot2grid(shape=(4, 2), loc=(1, 0), colspan=1)
    ax3 = plt.subplot2grid(shape=(4, 2), loc=(1, 1), rowspan=1)
    ax4 = plt.subplot2grid(shape=(4, 2), loc=(2, 0), rowspan=1)
    ax5 = plt.subplot2grid(shape=(4, 2), loc=(2, 1), rowspan=1)
    ax6 = plt.subplot2grid(shape=(4, 2), loc=(3, 0), rowspan=1)
    ax7 = plt.subplot2grid(shape=(4, 2), loc=(3, 1), rowspan=1)

    number_of_zeros = 50
    number_of_filter = 200

    zeros = np.zeros(number_of_zeros)
    filter1 = np.zeros(number_of_zeros)
    filter2 = np.zeros(number_of_zeros)
    filter3 = np.zeros(number_of_zeros)

    rec = np.ones(number_of_filter)
    filter1 = np.append(filter1, rec)
    filter1 = np.append(filter1, zeros)

    trian = signal.triang(number_of_filter)
    filter2 = np.append(filter2, trian)
    filter2 = np.append(filter2, zeros)

    hanning_window = np.hanning(number_of_filter)
    filter3 = np.append(filter3, hanning_window)
    filter3 = np.append(filter3, zeros)

    xscale = np.linspace(0, 3, 300)

    ax1.plot(x, y)
    ax2.plot(xscale, filter1)
    ax3.plot(x, np.multiply(filter1, y))
    ax4.plot(xscale, filter2)
    ax5.plot(x, np.multiply(filter2, y))
    ax6.plot(xscale, filter3)
    ax7.plot(x, np.multiply(filter3, y))

    axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]
    letters = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)']

    for i, ax in enumerate(axs):
        ax.set_ylabel("Amplituda")
        ax.set_xlabel("Czas [s]")
        if i == 0:
            ax.text(-0.10, 0.5, s=letters[i], transform=ax.transAxes, va='top', ha='right')
        ax.text(-0.22, 0.5, s=letters[i], transform=ax.transAxes, va='top', ha='right')

    # plt.tight_layout()
    plt.subplots_adjust(left=0.1,
                        bottom=0.05,
                        right=0.9,
                        top=0.95,
                        wspace=0.3,
                        hspace=0.3)
    plt.show()


def overlapping():
    directory = os.getcwd()
    audio_path = directory + '\Audio\\superstition.wav'
    audio_data, sr = librosa.load(audio_path)

    audio_data = audio_data[:88200]

    x_scale = np.linspace(0, 4, sr * 4)

    ones = np.hanning(sr)
    zeros = np.zeros(sr)

    # filter1 = np.concatenate((ones, zeros, zeros, zeros), axis=None)
    # filter2 = np.concatenate((zeros, ones, zeros, zeros), axis=None)
    # filter3 = np.concatenate((zeros, zeros, ones, zeros), axis=None)

    filter1 = np.concatenate((zeros, ones, zeros, zeros), axis=None)
    filter2 = np.concatenate((zeros, zeros, ones, zeros), axis=None)
    filter3 = np.concatenate((zeros, zeros, zeros, ones), axis=None)

    fig, ax = plt.subplots(4, 1, figsize=(10, 10))

    # ax[0].text(-0.12, 0.5, s='a)', transform=ax[0].transAxes, va='top', ha='right')
    # ax[0].legend(loc='upper right')
    # ax[0].set_ylabel("Amplituda")
    ax[0].plot(x_scale, audio_data)
    ax[0].set_xlabel("Czas [s]")
    ax[0].set_ylabel("Amplituda")
    ax[0].set_ylim(-1, 1)
    ax[0].grid()

    ax[1].plot(x_scale, np.multiply(audio_data, filter1))
    ax[1].plot(x_scale, filter1, 'r')
    ax[1].set_xlabel("Czas [s]")
    ax[1].set_ylabel("Amplituda")
    # ax[1].set_ylim(-1, 1)
    ax[1].grid()

    ax[2].plot(x_scale, np.multiply(audio_data, filter2))
    ax[2].plot(x_scale, filter2, 'r')
    ax[2].set_xlabel("Czas [s]")
    ax[2].set_ylabel("Amplituda")
    # ax[2].set_ylim(-1, 1)
    ax[2].grid()

    ax[3].plot(x_scale, np.multiply(audio_data, filter3))
    ax[3].plot(x_scale, filter3, 'r')
    ax[3].set_xlabel("Czas [s]")
    ax[3].set_ylabel("Amplituda")
    # ax[3].set_ylim(-1, 1)
    ax[3].grid()

    plt.tight_layout()

    plt.show()

    # -------------------------------------------------------------------------------------------------------

    filter1 = np.concatenate((zeros, ones, zeros, zeros), axis=None)
    filter2 = np.concatenate((zeros, zeros[:11025], ones, zeros, zeros[:11025]), axis=None)
    filter3 = np.concatenate((zeros, zeros, ones, zeros), axis=None)

    # filter1 = np.concatenate((ones, zeros, zeros, zeros), axis=None)
    # filter2 = np.concatenate((zeros[:11025], ones, zeros, zeros, zeros[:11025]), axis=None)
    # filter3 = np.concatenate((zeros, ones, zeros, zeros), axis=None)

    hop_size1 = np.linspace(1, 1.5, int(sr / 2))
    hop_size2 = np.linspace(1.5, 2, int(sr / 2))

    fig, ax = plt.subplots(4, 1, figsize=(10, 10))

    ax[0].plot(x_scale, audio_data)
    ax[0].set_xlabel("Czas [s]")
    ax[0].set_ylabel("Amplituda")
    ax[0].set_ylim(-1, 1)
    ax[0].grid()

    ax[1].plot(x_scale, np.multiply(audio_data, filter1))
    ax[1].plot(x_scale, filter1, 'r')
    ax[1].set_xlabel("Czas [s]")
    ax[1].set_ylabel("Amplituda")
    # ax[1].set_ylim(-1, 1)
    ax[1].grid()

    ax[2].plot(x_scale, np.multiply(audio_data, filter2))
    ax[2].plot(x_scale, filter2, 'r')
    ax[2].plot(hop_size1, np.ones(int(sr / 2)) / 2, 'k')
    ax[2].set_xlabel("Czas [s]")
    ax[2].set_ylabel("Amplituda")
    # ax[2].set_ylim(-1, 1)
    ax[2].grid()

    ax[2].annotate('długość skoku', xy=(0.95, 0.5), xytext=(0.25, 0.25),
                   arrowprops=dict(facecolor='black', width=0.5, headwidth=5, headlength=5),
                   )

    ax[3].plot(x_scale, np.multiply(audio_data, filter3))
    ax[3].plot(x_scale, filter3, 'r')
    ax[3].plot(hop_size2, np.ones(int(sr / 2)) / 2, 'k')
    ax[3].set_xlabel("Czas [s]")
    ax[3].set_ylabel("Amplituda")
    # ax[3].set_ylim(-1, 1)
    ax[3].grid()

    ax[3].annotate('długość skoku', xy=(1.45, 0.5), xytext=(0.75, 0.25),
                   arrowprops=dict(facecolor='black', width=0.5, headwidth=5, headlength=5),
                   )

    plt.tight_layout()
    plt.show()


def fourier():
    SAMPLE_RATE = 44100
    DURATION = 1
    FREQUENCY_1 = 5
    FREQUENCY_2 = 20
    FREQUENCY_3 = 50

    x1, y1 = generate_sine_wave(FREQUENCY_1, SAMPLE_RATE, DURATION, amplitude=1)
    _, y2 = generate_sine_wave(FREQUENCY_2, SAMPLE_RATE, DURATION, amplitude=0.5)
    _, y3 = generate_sine_wave(FREQUENCY_3, SAMPLE_RATE, DURATION, amplitude=0.25)

    y = y1 + y2 + y3
    N = SAMPLE_RATE * DURATION

    yf = rfft(y1 + y2 + y3)
    xf = rfftfreq(N, 1 / SAMPLE_RATE)
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    ax[0].plot(x1, y, 'g', label='sin(5Hz)+$\\frac{1}{2}$sin(20Hz)+$\\frac{1}{4}$sin(50Hz)')
    ax[0].text(-0.12, 0.5, s='a)', transform=ax[0].transAxes, va='top', ha='right')
    ax[0].legend(loc='upper right')
    ax[0].set_xlabel("Czas [s]")
    ax[0].set_ylabel("Amplituda")
    ax[0].set_ylim(-2, 2.5)
    ax[0].grid()

    ax[1].stem(xf[:100], np.abs(yf)[:100], 'k', basefmt="-k",
               label='widmo sygnału sin(5Hz)+$\\frac{1}{2}$sin(20Hz)+$\\frac{1}{4}$sin(50Hz)')
    ax[1].text(-0.12, 0.5, s='b)', transform=ax[1].transAxes, va='top', ha='right')
    ax[1].legend(loc='upper right')
    ax[1].set_xlabel("Częstotliwość [Hz]")
    ax[1].set_ylabel("Moduł wartości DFT")
    ax[1].grid()

    plt.subplots_adjust(
        bottom=0.1,
        top=0.95)

    plt.show()


def spectrogram():
    directory = os.getcwd()
    audio_path = directory + '\Audio\\superstition.wav'
    audio_data, sr = librosa.load(audio_path)

    audio_data = audio_data[:88200]
    data_stft = np.abs(librosa.stft(audio_data))
    data_stft_db = librosa.amplitude_to_db(data_stft)

    fig, ax = plt.subplots(figsize=(8, 6))
    # fig, ax = plt.subplots()

    img = librosa.display.specshow(data_stft_db, x_axis='time', y_axis='log', ax=ax)
    # ax.set_title(f"Power spectrogram\nFile:{self.audio_name}")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_xlabel("Czas [s]")
    ax.set_ylabel("Częstotliwość [Hz]")

    plt.subplots_adjust(
        bottom=0.1,
        top=0.95)
    plt.show()


def plot_harmonics():
    directory = os.getcwd()
    audio_name_piano = '\Audio\piano\single notes\piano A4.wav'
    audio_name_sax = '\Audio\sax\single notes\sax F#2.wav'

    audio_path_piano = directory + audio_name_piano
    audio_path_sax = directory + audio_name_sax

    y_sax, _ = librosa.load(path=audio_path_sax, sr=SAMPLE_RATE)
    y_piano, _ = librosa.load(path=audio_path_piano, sr=SAMPLE_RATE)

    ft_sax = sp.fft.fft(y_sax)
    magnitude_sax = np.absolute(ft_sax)
    frequency_sax = np.linspace(0, SAMPLE_RATE, len(magnitude_sax))

    ft_piano = sp.fft.fft(y_piano)
    magnitude_piano = np.absolute(ft_piano)
    frequency_piano = np.linspace(0, SAMPLE_RATE, len(magnitude_piano))

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    ax[0].plot(frequency_piano[:8000], magnitude_piano[:8000], label="Widmo Fouriera dźwięku A4 (pianino)")
    ax[0].text(-0.12, 0.5, s='a)', transform=ax[0].transAxes, va='top', ha='right')
    ax[0].legend(loc='upper right')
    ax[0].set_xlabel("Częstotliwość [Hz]")
    ax[0].set_ylabel("Moduł wartości DFT")
    ax[0].grid()

    ax[1].plot(frequency_sax[:8000], magnitude_sax[:8000], label="Widmo Fouriera dźwięku A4 (saksofon altowy)")
    ax[1].text(-0.12, 0.5, s='b)', transform=ax[1].transAxes, va='top', ha='right')
    ax[1].legend(loc='upper right')
    ax[1].set_xlabel("Częstotliwość [Hz]")
    ax[1].set_ylabel("Moduł wartości DFT")
    ax[1].grid()

    plt.subplots_adjust(
        bottom=0.1,
        top=0.95)

    plt.show()


def plot_spectral_flux():
    directory = os.getcwd()
    audio_name = '\Audio\piano\A blues scale.wav'

    top_db = 30
    audio_path = directory + audio_name
    audio_path = audio_path
    y, sr = librosa.load(audio_path, SAMPLE_RATE)
    y = y[320000:550000]
    o_env = librosa.onset.onset_strength(y, sr=sr, aggregate=np.mean)
    times = librosa.times_like(o_env, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)
    print("onset frames:\n", onset_frames)

    fig, ax = plt.subplots(nrows=2, sharex=True)
    librosa.display.waveshow(y, ax=ax[0], sr=sr, label='sygnał wejściowy')
    # ax[0].set_xlabel("Czas [s]")
    ax[0].legend(prop={'size': 8})
    ax[0].set_ylabel("Amplituda")
    # ax[0].text(-0.1, 0.5, s='a)', transform=ax[0].transAxes, va='top', ha='right')
    ax[1].plot(times, o_env, label='strumień widmowy')
    ax[1].vlines(times[onset_frames], 0, o_env.max(), color='r', alpha=0.9,
                 linestyle='--', label='wykryte miejsca'
                                       '')
    # ax[1].text(-0.1, 0.5, s='b)', transform=ax[1].transAxes, va='top', ha='right')
    ax[1].legend(prop={'size': 8})
    ax[1].set_xlabel("Czas [s]")
    ax[1].set_ylabel("Amplituda")

    plt.show()


def plot_f0_detection_steps():
    directory = os.getcwd()
    audio_name_piano = '\Audio\piano\single notes\piano A4.wav'

    audio_path_piano = directory + audio_name_piano

    # ax[0, 0] data
    signal, sr = librosa.load(path=audio_path_piano, sr=SAMPLE_RATE)
    frame_size = signal.shape[0]
    # ax[0, 1] data
    dt = 1 / sr
    freq_vector = np.fft.rfftfreq(frame_size, d=dt)
    windowed_signal = np.hamming(frame_size) * signal
    X = np.abs(np.fft.rfft(windowed_signal) / signal.shape[0])
    log_X = np.log(X)
    #
    df = freq_vector[1] - freq_vector[0]
    cepstrum = np.fft.rfft(log_X)
    # cepstrum = np.fft.rfft(X)

    quefrency_vector = np.fft.rfftfreq(log_X.size, df)

    # -----------------------------------------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))

    librosa.display.waveshow(signal, ax=ax[0, 0], sr=SAMPLE_RATE, label='x(t)')
    # ax[0, 0].legend(prop={'size': 8})
    # ax[0, 0].set_xlabel("")
    # ax[0, 0].text(-0.12, 0.5, s='a)', transform=ax[0, 0].transAxes, va='top', ha='right')
    ax[0, 0].legend(loc='upper right')
    ax[0, 0].set_xlabel("Czas [s]")
    ax[0, 0].grid()
    ax[0, 0].set_title('Analizowany sygnał')
    ax[0, 0].set_ylabel("Amplituda")

    ax[0, 1].plot(freq_vector[:8000], X[:8000], label="|X(f)|")
    # ax[0, 1].text(-0.12, 0.5, s='b)', transform=ax[0, 1].transAxes, va='top', ha='right')
    ax[0, 1].legend(loc='upper right')
    ax[0, 1].set_xlabel("Częstotliwość [Hz]")
    ax[0, 1].set_title('Widmo sygnału')
    ax[0, 1].grid()
    ax[0, 1].set_ylabel("Moduł wartości DFT")

    ax[1, 0].plot(freq_vector[:8000], log_X[:8000], label="ln|X(f)|")
    # ax[1, 0].text(-0.12, 0.5, s='c)', transform=ax[1, 0].transAxes, va='top', ha='right')
    ax[1, 0].legend(loc='upper right')
    ax[1, 0].set_xlabel("Częstotliwość [Hz]")
    ax[1, 0].set_title('Logarytmiczne widmo sygnału')
    ax[1, 0].grid()
    ax[1, 0].set_ylabel("ln|X(f)|")

    fmin, fmax = 50, 2500
    valid = (quefrency_vector > 1 / fmax) & (quefrency_vector <= 1 / fmin)
    # collection = collections.BrokenBarHCollection.span_where(
    #     quefrency_vector, ymin=0, ymax=np.abs(cepstrum).max(), where=valid, facecolor='green', alpha=0.5,
    #     label='valid pitches')

    ax[1, 1].plot(quefrency_vector, np.abs(cepstrum), label="c(x(t))")
    ax[1, 1].vlines(1 / 450, 0, np.max(np.abs(cepstrum)), alpha=.4, lw=3, label='znalezione f0', color='r')
    # ax[1, 1].add_collection(collection)
    # ax[1, 1].text(-0.1, 0.5, s='d)', transform=ax[1, 1].transAxes, va='top', ha='right')
    ax[1, 1].legend(loc='upper right')
    ax[1, 1].set_xlabel("Czas [s]")
    ax[1, 1].set_title('Cepstrum')
    ax[1, 1].grid()

    ax[1, 1].set_xlim(15 / 10000, 30 / 10000)
    ax[1, 1].set_ylim(0, 5000)
    ax[1, 1].set_ylabel("$F^{-1}(ln|X(f)|)$")

    plt.subplots_adjust(
        bottom=0.05,
        top=0.95,
        left=0.08,
        right=0.92)

    plt.show()


def plot_cepstrum():
    directory = os.getcwd()
    audio_name_piano = '\Audio\piano\single notes\piano A4.wav'

    audio_path_piano = directory + audio_name_piano

    y, sr = librosa.load(path=audio_path_piano, sr=SAMPLE_RATE)

    dt = 1 / sr
    freq_vector = np.fft.rfftfreq(y.shape[0], d=dt)  # size = 44100
    df = freq_vector[1] - freq_vector[0]
    X = np.abs(np.fft.rfft(y))

    quefrency_vector = np.fft.rfftfreq(y.shape[0], df)
    print(quefrency_vector)
    print(len(quefrency_vector))
    spectrum = np.fft.fft(y)
    log_spectrum = np.log(np.abs(spectrum))
    cepstrum = np.fft.ifft(log_spectrum).real

    min_freq, max_freq = FREQUENCY_RANGE
    start = int(sr / max_freq)
    end = int(sr / min_freq)
    narrowed_cepstrum = cepstrum[start:end]
    peak_ix = narrowed_cepstrum.argmax()
    freq0 = sr / (start + peak_ix)
    f0_inv = 1 / freq0

    cepstrum = cepstrum[:44101]

    narrowed_quefrency = []
    for q in quefrency_vector:
        if (q > 1 / max_freq) & (q <= 1 / min_freq):
            narrowed_quefrency.append(q)

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    ax[0].plot(quefrency_vector, np.abs(cepstrum))
    ax[0].vlines(f0_inv, 0, np.max(np.abs(cepstrum)), lw=2, label='f0', color='r')
    ax[0].legend(loc='upper right')
    ax[0].set_xlabel("Quefrency [s]")

    ax[1].plot(narrowed_quefrency, np.abs(narrowed_cepstrum))
    ax[1].vlines(f0_inv, 0, np.max(np.abs(narrowed_cepstrum)), lw=2, label='f0', color='r')
    ax[1].legend(loc='upper right')
    ax[1].set_xlabel("Quefrency [s]")

    plt.show()


def plot_detected_and_real_onsets():
    directory = os.getcwd()
    # audio_name = '\Audio\piano\C ionian scale.wav'
    # audio_name = '\Audio\sax\\autumn leaves high(150 bpm).wav'
    # audio_name = '\Audio\sax\\C ionian scale sax.wav'
    audio_name = '\Audio\sax\\F# blues scale sax.wav'

    audio_path = directory + audio_name
    audio_path = audio_path
    y, sr = librosa.load(audio_path, SAMPLE_RATE)

    y, sr = librosa.load(audio_path, SAMPLE_RATE)
    top_db = 30
    y_2 = np.zeros(y.shape[0])
    y_splitted = librosa.effects.split(y, top_db=top_db)
    for sound in y_splitted:
        start_idx = sound[0]
        end_idx = sound[1]
        y_2[start_idx:end_idx] = y[start_idx:end_idx]

    D = np.abs(librosa.stft(y))
    fig, ax = plt.subplots(nrows=2, figsize=(16, 6), sharex=True)

    real_onsets_times = [0.5, 0.832, 1.02, 1.326, 1.508, 1.787, 2.0, 2.284, 2.77, 3.287, 3.745, 4.0, 4.282, 4.464,
                         4.793,
                         5.32, 5.77, 6.0,
                         6.28, 6.46, 6.77, 7.268, 7.768, 7.955, 8.131, 8.278, 8.5, 8.8, 8.96, 9.29]
    real_onsets_frames = librosa.time_to_frames(real_onsets_times, sr=sr)
    #
    o_env = librosa.onset.onset_strength(y_2, sr=sr, aggregate=np.median)

    times = librosa.times_like(o_env, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)
    onset_frames = list(onset_frames)
    times_of_onsets = librosa.frames_to_time(onset_frames, sr=sr)

    def filter_onset(onset_frames):
        print(f"Detected onset frames: (len: {len(onset_frames)})\n", onset_frames)
        onset_frames = np.array(onset_frames)
        distance_list = []
        amplitudes = o_env[onset_frames]
        deleted_frames = []
        for i in range(len(onset_frames) - 1):
            distance = onset_frames[i + 1] - onset_frames[i]
            distance_list.append(distance)
        len_of_onset_frames = len(onset_frames) - 1
        for i in range(len_of_onset_frames):
            if distance_list[i] < 10:
                if amplitudes[i + 1] > amplitudes[i]:
                    deleted_frames.append(onset_frames[i])
                else:
                    deleted_frames.append(onset_frames[i + 1])
        print(deleted_frames)
        new_onsets = np.setdiff1d(onset_frames, deleted_frames)
        removed_indexes = []
        for i in range(1, len(new_onsets) - 2):
            if o_env[new_onsets[i]] < np.mean(o_env[new_onsets[i - 1:i + 2]]) / 2:
                removed_indexes.append(i)
        # print(f"New onset frames: (len: {len(new_onsets)})\n", new_onsets)
        # print(f"mean_removed: (len: {len(removed_indexes)})\n", removed_indexes)

        return new_onsets, deleted_frames, removed_indexes

    new_onsets, deleted_frames, removed = filter_onset(onset_frames)
    new_onsets = np.array(new_onsets)
    new_amplitudes = o_env[new_onsets]
    final_onsets = np.delete(new_onsets, removed)

    ax[0].plot(times, o_env, label='Onset strength')
    # ax[0].vlines(times[onset_frames], 0, o_env.max(), color='r', alpha=0.9, linestyle='--', label='Onsets')
    ax[0].vlines(times[final_onsets], 0, o_env.max(), color='r', alpha=0.9, linestyle='--', label='Final Onsets')

    # ax[0].vlines(times[real_onsets_frames], 0, o_env.max(), color='k', alpha=0.9, linestyle='--', label='Real Onsets')
    ax[0].legend()
    ax[0].set_xlabel("Czas [s]")

    ax[1].plot(times, o_env, label='Onset strength')
    ax[1].vlines(times[new_onsets], 0, o_env.max(), color='r', alpha=0.9, linestyle='--', label='New onsets')
    # ax[1].vlines(times[real_onsets_frames], 0, o_env.max(), color='k', alpha=0.9, linestyle='--', label='Real onsets')
    ax[1].legend()
    ax[1].set_xlabel("Czas [s]")

    plt.show()


def high_pass():
    b, a = signal.butter(4, 100, 'high', analog=True)
    w, h = signal.freqs(b, a)
    plt.semilogx(w, 20 * np.log10(abs(h)))
    # plt.title('Filtr Butterwortha')
    plt.xlabel('Częstotliwość [Hz]')
    plt.ylabel('Amplituda [dB]')
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.axvline(100, color='green')  # cutoff frequency
    plt.show()


def plot_accuracy():
    sax_ac = [0.88, 0.91, 0.88, 0.79, 0.85, 0.79, 0.73, 0.7, 0.67, 0.7, 0.64, 0.58, 0.64, 0.61, 0.61]
    piano_ac = [0.77, 0.85, 0.93, 0.98, 1., 1., 1., 1., 1., 1., 0.98, 0.98, 0.98, 0.97, 0.98]
    window_sizes = np.arange(1024, 8704, 512)
    fig, ax = plt.subplots()
    ax.plot(window_sizes[0::2], sax_ac[0::2], label="saksofon")
    ax.plot(window_sizes[0::2], piano_ac[0::2], label="pianino")
    ax.set_xticks(window_sizes[0::2])
    ax.legend()
    ax.set_xlabel("window_size")
    ax.set_ylabel("Skuteczność")
    plt.show()
    for i in range(len(sax_ac)):
        print(window_sizes[i], " & ", sax_ac[i], " \\")
    print()
    for i in range(len(piano_ac)):
        print(window_sizes[i], " & ", piano_ac[i], " \\")


if __name__ == "__main__":
    # plot_quantization()
    # plot_aliasing()
    # dtf_leakage()
    # window_functions()
    # overlapping()
    # fourier()
    # spectrogram()
    # plot_harmonics()
    # plot_spectral_flux()
    # plot_f0_detection_steps()
    # plot_cepstrum()
    # high_pass()
    # plot_accuracy()
    pass

