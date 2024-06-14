import essentia.standard as ess
import sys
sys.path.append('./datasets/traditional-flute-dataset/')
import numpy as np
import matplotlib.pyplot as plt
import csv



spectrum = ess.Spectrum()
w = ess.Windowing(type='hann')
hopsize = 128
framesize = 8192
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
    audio_filename = './datasets/Kaggle/traditional-flute-dataset/audio/allemande_fifth_fragment_preston.wav'
    # audio_filename = './datasets/GOOD-SOUNDS/good-sounds/sound_files/flute_almudena_dynamics_change/neumann/0000.wav'

    audio_loader = ess.MonoLoader(filename=audio_filename)
    audio = audio_loader()

    # uncomment for cleaning the flute audio
    # audio = audio[100000:300000]

    # uncomment to plot the audio signal
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    ax.plot(audio)
    ax.set_title('Audio')
    plt.show()

    fig.savefig("./images/audio.png")

    # uncomment to apply equal loudness
    # eq_loudness = ess.EqualLoudness()
    # audio = eq_loudness(audio)

    melodia = ess.PitchMelodia(frameSize=framesize, hopSize=hopsize)
    pitches_melodia = compute_pitches_no_frames(audio, melodia)

    pitches_yin = []
    pitches_yinfft = []
    pitches_hps = []

    for frame in ess.FrameGenerator(audio, frameSize=framesize, hopSize=hopsize):
        pitches_yin.append(pitchyin(frame, framesize))
        pitches_yinfft.append(pitchyinfft(frame, framesize))
        pitches_hps.append(hps(frame, framesize))

    old_indices = np.linspace(0, 1, len(pitches_melodia))
    new_indices = np.linspace(0, 1, len(pitches_yin))

    pitches_melodia = np.interp(new_indices, old_indices, pitches_melodia)

    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    ax.set_title(f'Estimated pitch depending on the algorithm')
    ax.plot(pitches_melodia, label='Melodia')
    ax.plot(pitches_yin, label='Yin' )
    ax.plot(pitches_yinfft, label='YinFFT')
    ax.plot(pitches_hps, label='HPS')
    ax.set_xlabel('Frames')
    ax.set_ylabel('Estimated pitch')
    ax.set_xlim(1000, 2000)
    ax.set_ylim(0, 1000)
    ax.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig("./images/estimated_pitch.png")



if __name__ == "__main__":
    main()
