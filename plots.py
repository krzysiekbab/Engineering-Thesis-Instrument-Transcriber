import matplotlib.pyplot as plt  # to plot
import numpy as np
from scipy.fft import fft, fftfreq, rfft, rfftfreq
from scipy import signal


def generate_sine_wave(freq, sample_rate, duration):
    x = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    frequencies = x * freq
    # 2pi because np.sin takes radians
    y = np.sin((2 * np.pi) * frequencies)
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

    rec_2 = np.zeros(50)
    rec_2 = np.append(rec_2, rec)
    rec_2 = np.append(rec_2, zeros)

    ax1.plot(x, y)
    ax2.plot(rec_2)
    ax3.plot(x, np.multiply(filter1, y))
    ax4.plot(trian)
    ax5.plot(x, np.multiply(filter2, y))
    ax6.plot(hanning_window)
    ax7.plot(x, np.multiply(filter3, y))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # plot_quantization()
    # plot_aliasing()
    # dtf_leakage2()
    # dtf_leakage()
    window_functions()
