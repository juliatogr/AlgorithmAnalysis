import copy

import essentia.standard as ess
import sys

sys.path.append('./datasets/Kaggle/traditional-flute-dataset/')
import numpy as np
import csv
import load as ld
from mir_eval import melody as mel

spectrum = ess.Spectrum()
w = ess.Windowing(type='hann')
hopsize = 128
framesize = 2048
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
    metrics = ['Voicing Recall', 'Voicing False Alarm', 'Raw Pitch Accuracy', 'Overall Accuracy']
    header = copy.deepcopy(metrics)
    header.insert(0, 'Audio Fragment')

    melodia_data = [header]
    yin_data = [header]
    yinfft_data = [header]
    hps_data = [header]

    fragments = ld.list_of_fragments('./datasets/Kaggle/traditional-flute-dataset/dataset.csv')

    for fragment in fragments:

        # load files: audio, gt
        audio_file = './datasets/Kaggle/traditional-flute-dataset/audio/' + fragment + '.wav'
        audio, t, fs = ld.audio(audio_file)

        gt_file = './datasets/Kaggle/traditional-flute-dataset/ground_truth/' + fragment + '.gt'
        gt_onset, gt_note, gt_duration = ld.ground_truth(gt_file)
        gt_array, gt_t, gt_index = ld.to_array(gt_onset, gt_note, gt_duration, fs, hopsize)

        # compute pitches
        audio_loader = ess.MonoLoader(filename=audio_file)
        audio = audio_loader()

        # uncomment to apply equal loudness
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

        pitches_yin = np.array(pitches_yin)
        pitches_yinfft = np.array(pitches_yinfft)
        pitches_hps = np.array(pitches_hps)

        # compute times for pitches
        times = np.arange(len(gt_array)) * hopsize / sample_rate

        # Interpolate pitches to match gt
        old_indices = np.linspace(0, 1, len(pitches_melodia))
        new_indices = np.linspace(0, 1, len(gt_array))

        pitches_melodia = np.interp(new_indices, old_indices, pitches_melodia)
        pitches_yin = np.interp(new_indices, old_indices, pitches_yin)
        pitches_yinfft = np.interp(new_indices, old_indices, pitches_yinfft)
        pitches_hps = np.interp(new_indices, old_indices, pitches_hps)

        try:
            melodia_metrics = mel.evaluate(gt_t, gt_array, times, pitches_melodia)
            yin_metrics = mel.evaluate(gt_t, gt_array, times, pitches_yin)
            yinfft_metrics = mel.evaluate(gt_t, gt_array, times, pitches_yinfft)
            hps_metrics = mel.evaluate(gt_t, gt_array, times, pitches_hps)

            melodia_data.append(
                [fragment, format_number(melodia_metrics[metrics[0]]), format_number(melodia_metrics[metrics[1]]),
                 format_number(melodia_metrics[metrics[2]]), format_number(melodia_metrics[metrics[3]])])
            yin_data.append(
                [fragment, format_number(yin_metrics[metrics[0]]), format_number(yin_metrics[metrics[1]]),
                 format_number(yin_metrics[metrics[2]]), format_number(yin_metrics[metrics[3]])])
            yinfft_data.append(
                [fragment, format_number(yinfft_metrics[metrics[0]]), format_number(yinfft_metrics[metrics[1]]),
                 format_number(yinfft_metrics[metrics[2]]), format_number(yinfft_metrics[metrics[3]])])
            hps_data.append(
                [fragment, format_number(hps_metrics[metrics[0]]), format_number(hps_metrics[metrics[1]]),
                 format_number(hps_metrics[metrics[2]]), format_number(hps_metrics[metrics[3]])])


        except Exception as e:
            print(f'Error in {fragment}: {e}')

        csv_file = f'./statistics/melodia_stats.csv'
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(melodia_data)

        csv_file = f'./statistics/yin_stats.csv'
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(yin_data)

        csv_file = f'./statistics/yinfft_stats.csv'
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(yinfft_data)

        csv_file = f'./statistics/hps_stats.csv'
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(hps_data)


def format_number(value):
    return f'{value:.3f}'.replace('.', ',')


if __name__ == "__main__":
    main()
