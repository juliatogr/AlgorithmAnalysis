import essentia.standard as ess
import sys

import matplotlib.pyplot as plt

sys.path.append('./datasets/traditional-flute-dataset/')
import time


spectrum = ess.Spectrum()
w = ess.Windowing(type='hann')
hopsize = 128
framesize = 4096
sample_rate = 44100
num_hps = 5  # max number of harmonic product spectrums

def get_rtf(extractor, audio, frame_splitting):
    start_time = time.time()

    if frame_splitting:
        pitches = []
        for frame in ess.FrameGenerator(audio, frameSize=framesize, hopSize=hopsize):
            pitch, confidence = extractor(frame)
            pitches.append(pitch)
    else:
        pitches, _ = extractor(audio)

    end_time = time.time()
    execution_time = end_time - start_time
    dur = len(audio) / sample_rate
    return execution_time / dur


def main():
    audio_filename = './datasets/Kaggle/traditional-flute-dataset/audio/allemande_fifth_fragment_preston_resampled.wav'

    audio_loader = ess.MonoLoader(filename=audio_filename)
    audio = audio_loader()

    rtf_melodia = get_rtf(ess.PitchMelodia(frameSize=framesize, hopSize=hopsize), audio, False)
    rtf_yin = get_rtf(ess.PitchYin(), audio, True)
    rtf_yinfft = get_rtf(ess.PitchYinFFT(), audio, True)
    rtf_hps = get_rtf(ess.PitchHPS(), audio, True)

    print(f"RTF Melodia: {rtf_melodia:.4f}")
    print(f"RTF Yin: {rtf_yin:.4f}")
    print(f"RTF YinFFT: {rtf_yinfft:.4f}")
    print(f"RTF HPS: {rtf_hps:.4f}")


if __name__ == "__main__":
    main()
