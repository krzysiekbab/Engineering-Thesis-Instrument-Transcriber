import matplotlib.pyplot as plt  # to plot
import numpy as np
from scipy.fft import fft, fftfreq, rfft, rfftfreq
from scipy import signal
import wave
import os

import librosa
import librosa.display


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
    ax[0].set_ylim(-1.5, 2)
    ax[0].grid()

    ax[1].plot(t, x1, 'g', label='Sygnał sinusoidalny o częstotliwości 50 [Hz]')
    ax[0].text(-0.12, 0.5, s='b)', transform=ax[1].transAxes, va='top', ha='right')
    ax[1].stem(nT, x2, 'o', label='Sygnał spróbkowany')
    ax[1].legend(loc='upper right')
    ax[1].set_xlabel("Czas [s]")
    ax[1].set_ylabel("Amplituda")
    ax[1].legend(loc='upper right')
    ax[1].set_ylim(-1.5, 2)
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
        ax.set_xlabel("Czas [s]")
        if i == 0:
            ax.text(-0.07, 0.5, s=letters[i], transform=ax.transAxes, va='top', ha='right')
        ax.text(-0.15, 0.5, s=letters[i], transform=ax.transAxes, va='top', ha='right')

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

    fig, ax = plt.subplots(4, 1, figsize=(10, 8))

    # ax[0].text(-0.12, 0.5, s='a)', transform=ax[0].transAxes, va='top', ha='right')
    # ax[0].legend(loc='upper right')
    # ax[0].set_ylabel("Amplituda")
    ax[0].plot(x_scale, audio_data)
    ax[0].set_xlabel("Czas [s]")
    ax[0].set_ylim(-1, 1)
    ax[0].grid()

    ax[1].plot(x_scale, np.multiply(audio_data, filter1))
    ax[1].plot(x_scale, filter1, 'r')
    ax[1].set_xlabel("Czas [s]")
    ax[1].grid()

    ax[2].plot(x_scale, np.multiply(audio_data, filter2))
    ax[2].plot(x_scale, filter2, 'r')
    ax[2].set_xlabel("Czas [s]")
    ax[2].grid()

    ax[3].plot(x_scale, np.multiply(audio_data, filter3))
    ax[3].plot(x_scale, filter3, 'r')
    ax[3].set_xlabel("Czas [s]")
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

    fig, ax = plt.subplots(4, 1, figsize=(10, 8))

    ax[0].plot(x_scale, audio_data)
    ax[0].set_xlabel("Czas [s]")
    ax[0].set_ylim(-1, 1)
    ax[0].grid()

    ax[1].plot(x_scale, np.multiply(audio_data, filter1))
    ax[1].plot(x_scale, filter1, 'r')
    ax[1].set_xlabel("Czas [s]")
    ax[1].grid()

    ax[2].plot(x_scale, np.multiply(audio_data, filter2))
    ax[2].plot(x_scale, filter2, 'r')
    ax[2].plot(hop_size1, np.ones(int(sr / 2)) / 2, 'k')
    ax[2].set_xlabel("Czas [s]")
    ax[2].grid()

    ax[2].annotate('długość skoku', xy=(0.95, 0.5), xytext=(0.25, 0.25),
                   arrowprops=dict(facecolor='black', width=0.5, headwidth=5, headlength=5),
                   )

    ax[3].plot(x_scale, np.multiply(audio_data, filter3))
    ax[3].plot(x_scale, filter3, 'r')
    ax[3].plot(hop_size2, np.ones(int(sr / 2)) / 2, 'k')
    ax[3].set_xlabel("Czas [s]")
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

    N = SAMPLE_RATE * DURATION

    yf = rfft(y1 + y2 + y3)
    xf = rfftfreq(N, 1 / SAMPLE_RATE)
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    ax[0].plot(x1, y1 + y2 + y3, 'g', label='sin(5Hz)+$\\frac{1}{2}$sin(20Hz)+$\\frac{1}{4}$sin(50Hz)')
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


if __name__ == "__main__":
    # plot_quantization()
    # plot_aliasing()
    # dtf_leakage()
    # window_functions()
    # overlapping()
    # fourier()
    # spectrogram()
    pass
