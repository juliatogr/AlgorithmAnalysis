import essentia.standard as ess
import numpy as np
import matplotlib.pyplot as plt


spectrum = ess.Spectrum()
w = ess.Windowing(type='hann')

hopsize = 128
framesize = 4096
sample_rate = 44100
window_size = 44100
num_harmonics = 5


def hps(frame):
    spec = spectrum(w(frame))
    hps = ess.HarmonicProductSpectrum(frameSize=framesize, numHarmonics=num_harmonics)
    pitch, _ = hps(spec)
    return pitch


def generate_sine():
    duration = 1.5
    fundamental_freq = 132

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    fundamental = np.sin(2 * np.pi * fundamental_freq * t)
    second_harmonic = 0.7*np.sin(2 * np.pi * 2 * fundamental_freq * t)
    third_harmonic = 0.6* np.sin(2 * np.pi * 3 * fundamental_freq * t) / 4
    fourth_harmonic = 0.4*np.sin(2*np.pi*4*fundamental_freq * t) / 8

    combined_signal = fundamental + second_harmonic + third_harmonic + fourth_harmonic

    return np.array(combined_signal / np.max(np.abs(combined_signal)), dtype=np.float32)


def plot_audio(audio):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6.5))
    ax.plot(audio)
    plt.show()
    fig.savefig("sine_HPS.png")


def main():

    audio = generate_sine()
    # plot_audio(audio)

    num_hps = 4

    # hps_temp = ess.HarmonicProductSpectrum(frameSize=framesize, hopSize=hopsize, sampleRate=sample_rate, numHps=num_hps)
    # pitches_hps = hps_temp(audio[1000:1000+4*framesize])
    # print("*****Harmonic Product Spectrum*****")
    # print(f'    - pitches: {pitches_hps}')
    # print(f'    - length: {len(pitches_hps)}')
    # print(f'audio length: {len(audio[1000:1000+2*framesize])/framesize}')

    # f = np.arange(0, len(pitches_hps)) * sample_rate / (framesize)
    #
    # plt.plot(pitches_hps, label='HPS implemented')
    # plt.legend()
    # plt.show()

    # windowing = ess.Windowing(type='blackmanharris62', zeroPadding=2048)
    # spectrum = ess.Spectrum()
    #
    # amp2db = ess.UnaryOperator(type='lin2db', scale=2)
    # pool = es.Pool()
    #
    # for frame in ess.FrameGenerator(audio[1000:1000+2*framesize], frameSize=2048, hopSize=1024):
    #     frame_spectrum = spectrum(windowing(frame))
    #     pool.add('spectrum_db', amp2db(frame_spectrum))
    #
    # # Plot the spectrogram on ax1
    # fig, ax1 = plt.subplots(1, 1, figsize=(15, 8))
    #
    # ax1.set_title("Log-spectrogram (amp2db)")
    # ax1.set_xlabel("Time (frames)")
    # ax1.set_ylabel("Frequency bins")
    # ax1.set_ylim(0, 250)
    # ax1.imshow(pool['spectrum_db'].T, aspect='auto', origin='lower', interpolation='none')
    #
    # plt.show()
    #
    # print("frame size: ", framesize)
    # print("frame spectrum size: ", 2*framesize+1)
    # print("num_hps: ", num_hps)
    # print("freq pos: ", np.argmax(pitches_hps))

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12))

    spec = ess.Spectrum()
    sinespec = spec(audio[:10000])
    print("len sinespec",len(sinespec))
    sinespec[:25] = 0

    ax1.plot(sinespec)
    ax1.set_title("Original spectrum")
    ax1.set_xlim(0, 130)
    ax1.set_xlabel("Samples")
    ax1.set_ylabel("Amplitude")
    ax1.set_xticks(np.arange(0, 131, 15))

    max_val = max(sinespec)

    sinespec[sinespec < 20] = 0

    sinespec2 = np.zeros(len(sinespec))
    sinespec2[:sinespec.size//2+1] = sinespec[::2]

    # interpol = ess.Interpolator()
    ax2.plot(sinespec2)
    ax2.set_title("Downsampled by factor 2")

    ax2.set_xlim(0,130)
    ax2.set_xlabel("Samples")
    ax2.set_ylabel("Amplitude")
    ax2.set_xticks(np.arange(0, 131, 15))


    sinespec3 = np.zeros(len(sinespec))
    sinespec3[:sinespec.size//3] = sinespec[::3]

    ax3.plot(sinespec3)
    ax3.set_xlim(0, 130)
    ax3.set_title("Downsampled by factor 3")
    ax3.set_xlabel("Samples")
    ax3.set_ylabel("Amplitude")
    ax3.set_xticks(np.arange(0, 131, 15))

    mult = sinespec*sinespec2*sinespec3
    ax4.plot(mult)
    ax4.set_xlim(0, 130)
    ax4.set_title("HPS")
    ax4.set_xlabel("Samples")
    ax4.set_ylabel("Amplitude")
    ax4.set_xticks(np.arange(0, 131, 15))

    plt.tight_layout()
    plt.show()

    # save plot
    fig.savefig("sine_HPS.png")


if __name__ == "__main__":
    main()
