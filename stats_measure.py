import essentia.standard as ess
import sys
sys.path.append('./datasets/traditional-flute-dataset/')
import numpy as np
import matplotlib.pyplot as plt



spectrum = ess.Spectrum()
w = ess.Windowing(type='hann')
hopsize = 128
sample_rate = 44100
num_hps = 5


def hps(frame, framesize):
    spec = spectrum(w(frame))
    hps = ess.PitchHPS(frameSize=framesize, numHarmonics=num_hps)
    pitch, _ = hps(spec)
    return pitch


def pitchyinfft(frame, framesize):
    spec = spectrum(w(frame))
    pitchyinfft = ess.PitchYinFFT(frameSize=framesize)
    pitch, confidence = pitchyinfft(spec)
    return pitch


def pitchyin(frame, framesize):
    pitchyin = ess.PitchYin(frameSize=framesize)
    pitch, confidence = pitchyin(frame)
    return pitch


def compute_pitches_no_frames(audio, extractor):
    audio_pitches, _ = extractor(audio)
    return audio_pitches


def main():
    # audio_filename = './datasets/GOOD-SOUNDS/good-sounds/sound_files/clarinet_pablo_pitch_stability/neumann/0000.wav'
    audio_filename = './datasets/GOOD-SOUNDS/good-sounds/sound_files/flute_almudena_dynamics_change/akg/0000.wav'

    audio_loader = ess.MonoLoader(filename=audio_filename)
    audio = audio_loader()

    # uncomment for cleaning the flute audio
    audio = audio[100000:300000]

    # plot spectrum of first frame
    # spec = spectrum(w(audio[:framesize]))
    # f = np.arange(0, len(spec)) * sample_rate / framesize
    # plt.plot(f, spec)
    # plt.xlim(0, 4500)
    # plt.show()

    # uncomment to apply equal loudness
    # eq_loudness = ess.EqualLoudness()
    # audio = eq_loudness(audio)

    framesizes = [2048, 4096, 8192]

    fig, axs = plt.subplots(3, 1, figsize=(15, 12))
    for i in range(len(framesizes)):
        print(f'Stats for framesize {framesizes[i]} \n')

        melodia = ess.PitchMelodia(frameSize=framesizes[i], hopSize=hopsize)
        pitches_melodia = compute_pitches_no_frames(audio, melodia)

        predominant_melodia = ess.PredominantPitchMelodia(frameSize=framesizes[i], hopSize=hopsize)
        pitches_predominant_melodia = compute_pitches_no_frames(audio, predominant_melodia)

        pitches_yin = []
        pitches_yinfft = []
        pitches_hps = []

        for frame in ess.FrameGenerator(audio, frameSize=framesizes[i], hopSize=hopsize, startFromZero=True):
            pitches_yin.append(pitchyin(frame, framesizes[i]))
            pitches_yinfft.append(pitchyinfft(frame, framesizes[i]))
            pitches_hps.append(hps(frame, framesizes[i]))

        axs[i].set_title(f'Framesize {framesizes[i]}')
        axs[i].plot(pitches_yin, label='Yin' )
        axs[i].plot(pitches_yinfft, label='YinFFT')
        axs[i].plot(pitches_hps, label='HPS')
        axs[i].plot(pitches_melodia, label='Melodia')
        axs[i].plot(pitches_predominant_melodia, label='PredominantMelodia')
        axs[i].set_xlabel('Frames')
        axs[i].set_ylabel('Estimated pitch')
        axs[i].legend()

        print(f"Mean +- std melodia: {np.mean(pitches_melodia)} +- {np.std(pitches_melodia)}")
        print(f"Mean +- std predominantmelodia: {np.mean(pitches_predominant_melodia)} +- {np.std(pitches_predominant_melodia)}")
        print(f"Mean +- std yin: {np.mean(pitches_yin)} +- {np.std(pitches_yin)}")
        print(f"Mean +- std yinFFT: {np.mean(pitches_yinfft)} +- {np.std(pitches_yinfft)}")
        print(f"Mean +- std hps: {np.mean(pitches_hps)} +- {np.std(pitches_hps)}")

        print(f'\n\n')

    fig.title('Pitch estimation depending on framesize')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
