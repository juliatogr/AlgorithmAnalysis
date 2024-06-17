import essentia.standard as ess
import sys
sys.path.append('./datasets/Traditional Flute/')
import load as ld
import numpy as np
import matplotlib.pyplot as plt


spectrum = ess.Spectrum()
w = ess.Windowing(type='hann')
hopsize = 128
framesize = 4096
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
    audio_filename = './datasets/Traditional Flute/audio/syrinx_fifth_fragment_bourdin.wav'

    gt_file = './datasets/Traditional Flute/ground_truth/syrinx_fifth_fragment_bourdin.gt'
    gt_onset, gt_note, gt_duration = ld.ground_truth(gt_file)
    gt_array, gt_t, gt_index = ld.to_array(gt_onset, gt_note, gt_duration)
    audio_loader = ess.MonoLoader(filename=audio_filename)
    audio = audio_loader()

    # uncomment to plot the audio signal
    # fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    # ax.plot(audio)
    # ax.set_title('Audio')
    # plt.show()
    #
    # fig.savefig("./images/audio.png")

    eq_loudness = ess.EqualLoudness()
    cleaned_audio = eq_loudness(audio)

    melodia = ess.PitchMelodia(frameSize=framesize, hopSize=hopsize)
    pitches_melodia = compute_pitches_no_frames(cleaned_audio, melodia)

    pitches_yin = []
    pitches_yinfft = []
    pitches_hps = []

    for frame in ess.FrameGenerator(audio, frameSize=framesize, hopSize=hopsize):
        pitches_yin.append(pitchyin(frame, framesize))
        pitches_yinfft.append(pitchyinfft(frame, framesize))
        pitches_hps.append(hps(frame, framesize))

    # Interpolate pitches to match gt
    old_indices = np.linspace(0, 1, len(pitches_melodia))
    new_indices = np.linspace(0, 1, len(gt_array))

    pitches_melodia = np.interp(new_indices, old_indices, pitches_melodia)
    pitches_yin = np.interp(new_indices, old_indices, pitches_yin)
    pitches_yinfft = np.interp(new_indices, old_indices, pitches_yinfft)
    pitches_hps = np.interp(new_indices, old_indices, pitches_hps)

    times = np.arange(0, len(pitches_melodia)) * hopsize / sample_rate

    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    ax.set_title(f'Estimated pitch depending on the algorithm')
    ax.plot(times, pitches_melodia, label='Melodia', linestyle='--')
    ax.plot(times, pitches_yin, label='Yin', linestyle='--')
    ax.plot(times, pitches_yinfft, label='YinFFT', linestyle='--')
    ax.plot(times, pitches_hps, label='HPS')
    ax.plot(gt_t, gt_array, label='Reference')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pitch (Hz)')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1400)
    ax.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig("./images/estimated_pitch_start.png")

    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    ax.set_title(f'Estimated pitch depending on the algorithm')
    ax.plot(times, pitches_melodia, label='Melodia', linestyle='--')
    ax.plot(times, pitches_yin, label='Yin', linestyle='--')
    ax.plot(times, pitches_yinfft, label='YinFFT', linestyle='--')
    ax.plot(times, pitches_hps, label='HPS')
    ax.plot(gt_t, gt_array, label='Reference')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pitch (Hz)')
    ax.set_xlim(40, 50)
    ax.set_ylim(0, 600)
    ax.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig("./images/estimated_pitch_end.png")




if __name__ == "__main__":
    main()
