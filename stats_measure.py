import essentia.standard as ess
import sys
sys.path.append('./datasets/Kaggle/traditional-flute-dataset/')
import numpy as np
import matplotlib.pyplot as plt
import csv
import load as ld
import librosa as lr
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
    fragments = ld.list_of_fragments('./datasets/Kaggle/traditional-flute-dataset/dataset.csv')

    for fragment in fragments:

        # load files: audio, gt, score
        audio_file = './datasets/Kaggle/traditional-flute-dataset/audio/' + fragment + '.wav'
        audio, t, fs = ld.audio(audio_file)

        gt_file = './datasets/Kaggle/traditional-flute-dataset/ground_truth/' + fragment + '.gt'
        gt_onset, gt_note, gt_duration = ld.ground_truth(gt_file)
        gt_array, gt_t, gt_index = ld.to_array(gt_onset, gt_note, gt_duration, fs, hopsize)

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

        times = np.arange(len(gt_array)) * hopsize / sample_rate

        # Generate new x positions (indices) for array y
        old_indices = np.linspace(0, 1, len(pitches_melodia))
        new_indices = np.linspace(0, 1, len(gt_array))

        # Interpolate
        pitches_melodia = np.interp(new_indices, old_indices, pitches_melodia)
        pitches_yin = np.interp(new_indices, old_indices, pitches_yin)
        pitches_yinfft = np.interp(new_indices, old_indices, pitches_yinfft)
        pitches_hps = np.interp(new_indices, old_indices, pitches_hps)

        try:
            melodia_metrics = mel.evaluate(gt_t, gt_array, times, pitches_melodia)
            yin_metrics = mel.evaluate(gt_t, gt_array, times, pitches_yin)
            yinffy_metrics = mel.evaluate(gt_t, gt_array, times, pitches_yinfft)
            hps_metrics = mel.evaluate(gt_t, gt_array, times, pitches_hps)

            pitch_methods = [
                ('PitchMelodia', pitches_melodia, melodia_metrics),
                ('PitchYin', pitches_yin, yin_metrics),
                ('PitchYinFFT', pitches_yinfft, yinffy_metrics),
                ('PitchHPS', pitches_hps, hps_metrics),
                ('Ground-Truth', gt_array, None)
            ]

            data = [['Pitch Method', 'Mean', 'Standard Deviation']]
            for method, pitches, metrics in pitch_methods:
                data.append([method, format_number(np.mean(pitches)), format_number(np.std(pitches))])

            data.append([])

            data.append(
                ['Pitch Method', 'Voicing Recall', 'Voicing False Alarm', 'Raw Pitch Accuracy', 'Overall Accuracy'])
            for method, _, metrics in pitch_methods[:-1]:  # Exclude Ground-Truth from this part
                data.append([
                    method,
                    format_number(metrics['Voicing Recall']),
                    format_number(metrics['Voicing False Alarm']),
                    format_number(metrics['Raw Pitch Accuracy']),
                    format_number(metrics['Overall Accuracy'])
                ])
            csv_file = f'./statistics/{fragment}.csv'
            with open(csv_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(data)

        except Exception as e:
            print(f'Error in {fragment}: {e}')


def format_number(value):
    return f'{value:.3f}'.replace('.', ',')

if __name__ == "__main__":
    main()
