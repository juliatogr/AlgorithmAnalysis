import essentia as es
import essentia.standard as ess
import sys

sys.path.append('./datasets/traditional-flute-dataset/')
import load as ld

import mir_eval
import numpy as np
import time
import mido
import matplotlib.pyplot as plt
import math
import copy
import scipy.fftpack
import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import librosa as lr
from pysndfx import AudioEffectsChain


spectrum = ess.Spectrum()
w = ess.Windowing(type='hann')
hopsize = 128
framesize = 4096
sample_rate = 44100

window_size = 44100  # window size of the DFT in samples

num_hps = 5  # max number of harmonic product spectrums
power_th = 1e-6  # tuning is activated if the signal power exceeds this threshold
white_noise_th = 0.2  # everything under white_noise_th*avg_energy_per_freq is cut off

delta_freq = sample_rate / window_size  # frequency step width of the interpolated DFT
octave_bands = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]

hanning_window = np.hanning(window_size)

# Harmonic Product Spectrum (HPS) pitch detector



def hps(frame):
    fft = ess.FFT()
    spec = spectrum(w(frame))
    real_fft = fft(frame)
    plt.plot(real_fft[:len(real_fft) // 2])
    plt.show()
    hps = ess.HarmonicProductSpectrum(frameSize=framesize, numDownsamplings=num_hps)
    pitch, _ = hps(spec)
    return pitch


def pitchyinprobabilities(frame):
    pitchyinprobabilities = ess.PitchYinProbabilities(frameSize=framesize)
    pitches, probabilities, rms = pitchyinprobabilities(frame)
    return pitches[np.argmax(probabilities)]


def pitchyinfft(frame):
    spec = spectrum(w(frame))
    pitchyinfft = ess.PitchYinFFT(frameSize=framesize)
    pitch, confidence = pitchyinfft(spec)
    return pitch


def pitchyin(frame):
    pitchyin = ess.PitchYin(frameSize=framesize)
    pitch, confidence = pitchyin(frame)
    return pitch


def compute_pitches_no_frames(audio, extractor):
    audio_pitches, _ = extractor(audio)
    return audio_pitches


def load_midi(filename):
    midi_data = mido.MidiFile(filename)
    notes = []
    for msg in midi_data:
        if msg.type == 'note_on':
            pitch = msg.note
            time = msg.time
            notes.append((pitch, time))
    return notes


def align_pitches(audio_pitches, midi_notes):
    numsamples = len(audio_pitches)
    aligned_pitches = []
    aligned_times = np.arange(0, numsamples / sample_rate, 1 / sample_rate)
    for i in range(len(audio_pitches)):
        aligned_pitches.append(audio_pitches[i])

    print(aligned_pitches)
    print(len(aligned_times))
    return aligned_pitches, aligned_times


def main():
    # audio_filename = './datasets/traditional-flute-dataset/audio/allemande_fifth_fragment_preston_resampled.wav'
    audio_filename = './datasets/Bad recordings/C Scale 1.wav'
    # midi_filename = "datasets/generated_scales/C Major/C Scale .mid"
    #
    audio_loader = ess.MonoLoader(filename=audio_filename)
    audio = audio_loader()

    fragments = ld.list_of_fragments('./datasets/Kaggle/traditional-flute-dataset/dataset.csv')
    fragment = fragments[10]
    print('fragment: ' + fragment)

    duration = 1.5
    fundamental_freq = 132

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    fundamental = np.sin(2 * np.pi * fundamental_freq * t)
    second_harmonic = 0.7*np.sin(2 * np.pi * 2 * fundamental_freq * t)
    third_harmonic = 0.6* np.sin(2 * np.pi * 3 * fundamental_freq * t) / 4
    fourth_harmonic = 0.4*np.sin(2*np.pi*4*fundamental_freq * t) / 8

    combined_signal = fundamental + second_harmonic + third_harmonic + fourth_harmonic

    audio = np.array(combined_signal / np.max(np.abs(combined_signal)), dtype=np.float32)



    # # midi_notes = load_midi(midi_filename)
    # #
    # eq_loudness = ess.EqualLoudness()
    # cleaned_audio = eq_loudness(audio)

    # melodia = ess.PitchMelodia(frameSize=framesize, hopSize=hopsize)
    # pitches_melodia = compute_pitches_no_frames(audio[1000:1000+2*framesize], melodia)
    #
    # # aligned_pitches, aligned_times = align_pitches(pitches_melodia,  midi_notes)
    # #
    # # print("*****Midi*****")
    # # print(f'    - length {len(aligned_pitches)}')
    # # print(f'    - length {aligned_times}')
    #
    # # accuracy = mir_eval.melody.evaluate(np.asarray(midi_notes)[:,0], np.asarray(midi_notes)[:,1], np.asarray(aligned_pitches), np.asarray(aligned_times))
    # # print(f" Melodia Accuracy: {accuracy}")
    #
    # print("*****PitchMelodia*****")
    # # print(type(pitches_melodia)) #numpy.ndarray
    # print(f'    - pitches: {pitches_melodia}')
    # # print(f'    - length {len(pitches_melodia)}')

    # predominant_melodia = ess.PredominantPitchMelodia(frameSize=framesize, hopSize=hopsize)
    # pitches_predominant_melodia = compute_pitches_no_frames(audio, predominant_melodia)

    # print("*****PredominantPitchMelodia*****")
    # print(type(pitches_predominant_melodia)) #numpy.ndarray
    # print(f'    - pitches: {pitches_predominant_melodia}')
    # print(f'    - length {len(pitches_predominant_melodia)}')

    # pitches_yin = []
    # pitches_yinfft = []
    # pitches_yinprobabilities = []
    pitches_hps = []
    pitches_hps_2 = []


    # for frame in ess.FrameGenerator(audio, frameSize=framesize, hopSize=hopsize,  startFromZero=True):
    #     #    pitches_yin.append(pitchyin(frame))
    #     # pitches_yinfft.append(pitchyinfft(frame))
    #     #    pitches_yinprobabilities.append(pitchyinprobabilities(frame))
    #     # pitches_hps.append(hps(frame))
    #     pitches_hps_2.append(hps_pitch_detector(frame))

    # print("*****PitchYin*****")
    # print(type(pitches_yin)) #list
    # print(f'    - pitches: {pitches_yin}')
    # print(f'    - length: {len(pitches_yin)}')

    # print("*****PitchYinFFT*****")
    # # # print(type(pitches_yinfft)) #list
    # print(f'    - pitches: {pitches_yinfft}')
    # print(f'    - length: {len(pitches_yinfft)}')
    #
    # pitch_yin_probabilistic = ess.PitchYinProbabilistic(frameSize=framesize, hopSize=hopsize)
    # pitches_yinprobabilistic, _ = pitch_yin_probabilistic(audio)

    # print("*****PitchYinProbabilistic*****")
    # print(type(pitches_yinprobabilistic))  #numpy.ndarray
    # print(f'    - pitches: {pitches_yinprobabilistic}')
    # print(f'    - length: {len(pitches_yinprobabilistic)}')

    # print("*****PitchYinProbabilities*****")
    # print(type(pitches_yinprobabilities))  #list
    # print(f'    - pitches: {pitches_yinprobabilities}')
    # print(f'    - length: {len(pitches_yinprobabilities)}')

    end_time = time.time()
    execution_time = end_time - start_time

    audio_dur = len(audio)/sr
    rtf = execution_time/audio_dur




if __name__ == "__main__":
    main()
