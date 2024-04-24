import sys
sys.path.append('./datasets/traditional-flute-dataset/')
import fastdtw
import IPython.display
import numpy as np
import matplotlib.pyplot as plt
import librosa as lr
import load as ld

def note_filter(f0_array, fmin=lr.note_to_midi('B3'), fmax=lr.note_to_midi('B9'), fs=44100, resolution_octave=1, harmonics=0, beta=0.1):
    faux = np.linspace(fmin, fmax, (fmax-fmin)*resolution_octave)
    filter_aux = np.zeros((len(faux), len(f0_array)))
    for j in range(0, len(f0_array)):
        if f0_array[j] == 0:
            filter_aux[:,j] = beta*np.ones(len(faux))
        else:
            idx=resolution_octave*int(f0_array[j]-fmin)
            filter_aux[idx, j] = 1
            if harmonics > 0:
                if (idx+12*resolution_octave) < len(faux): #octava
                    filter_aux[idx+12*resolution_octave, j] = 1
                if harmonics > 1:
                    if (idx+19*resolution_octave) < len(faux): #octava + quinta
                        filter_aux[idx+19*resolution_octave, j] = 1
                        if harmonics > 2:
                            if (idx+24*resolution_octave) < len(faux): #octava + quinta
                                filter_aux[idx+24*resolution_octave, j] = 1
                            if harmonics > 3:
                                if (idx+28*resolution_octave) < len(faux): #octava + quinta
                                    filter_aux[idx+28*resolution_octave, j] = 1
                                if harmonics > 4:
                                    if (idx+31*resolution_octave) < len(faux): #octava + quinta
                                        filter_aux[idx+31*resolution_octave, j] = 1
                                    if harmonics > 5:
                                        if (idx+34*resolution_octave) < len(faux): #octava + quinta
                                            filter_aux[idx+31*resolution_octave, j] = 1

    return filter_aux

def main():
    # distance function param
    distancefun_param = ['cosine', 'euclidean']
    distancefun = distancefun_param[0]
    print('distance: ' + distancefun)

    # bins params
    bins_param = [12, 24, 36]
    bins = bins_param[1]
    print('bins per octave: ' + str(bins))

    # range param: flute register
    end_note = lr.note_to_midi('B9')
    range_bins = lr.note_to_midi('B9') - lr.note_to_midi('B3')

    # harmonics param
    harmonics_param = [0, 1, 2, 3, 4, 5, 6]
    harmonics = harmonics_param[1]
    print('harmonics in score codification: ' + str(harmonics))

    # windows params
    hop_param = [128, 256, 512, 1024, 2048, 4096]
    hop = hop_param[1]
    print('hop size: ' + str(hop))

    # fragment params
    fragments = ld.list_of_fragments('./datasets/traditional-flute-dataset/dataset.csv')
    fragment = fragments[10]
    print('fragment: ' + fragment)

    # load files: audio, gt, score
    audio_file = './datasets/traditional-flute-dataset/audio/' + fragment + '.wav'
    audio, t, fs = ld.audio(audio_file)

    gt_file = './datasets/traditional-flute-dataset/ground_truth/' + fragment + '.gt'
    gt_onset, gt_note, gt_duration = ld.ground_truth(gt_file)
    gt_note = lr.hz_to_midi(gt_note)
    gt_note[np.isinf(gt_note)] = 0
    gt_array, gt_t, gt_index = ld.to_array(gt_onset, gt_note, gt_duration, fs, hop)

    score_file = './datasets/traditional-flute-dataset/score/' + fragment[0:fragment.rfind('_')] + '.notes'
    score_onset, score_note, score_duration = ld.score(score_file)
    score_array, score_t, score_index = ld.to_array(score_onset, score_note, score_duration, fs, hop)

    # generate intermediate representation with score
    note_fb = note_filter(score_array, resolution_octave=int(bins / 12), harmonics=harmonics)

    C = lr.core.cqt(audio, sr=fs, hop_length=hop, fmin=lr.note_to_hz('B3'),
                    n_bins=range_bins * int(bins / 12), bins_per_octave=bins, tuning=None,
                    filter_scale=1, sparsity=0.3, norm=1, scale=True)

    Cxx = np.abs(C) + 0.001

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.pcolormesh(note_fb)
    plt.subplot(1, 2, 2)
    plt.pcolormesh(np.log10(Cxx))

if __name__ == "__main__":
    main()
