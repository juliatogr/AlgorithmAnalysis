import essentia.standard as ess

import mir_eval
import numpy as np
import time
import mido

spectrum = ess.Spectrum()
w = ess.Windowing(type='hann')
framesize = 2048
hopsize = 128


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



# Load MIDI file and extract pitch and timing information
def load_midi(filename):
    midi_data = mido.MidiFile(filename)
    notes = []
    for msg in midi_data:
        if msg.type == 'note_on':
            pitch = msg.note
            time = msg.time
            notes.append((pitch, time))
    return notes

# Align MIDI pitches with audio pitches based on timing information
def align_pitches(audio_pitches, audio_times, midi_notes):
    aligned_pitches = []
    aligned_times = []
    midi_times = np.cumsum([note[1] for note in midi_notes])
    midi_pitches = [note[0] for note in midi_notes]
    audio_time_index = 0
    for midi_time, midi_pitch in zip(midi_times, midi_pitches):
        while audio_time_index < len(audio_times) and audio_times[audio_time_index] < midi_time:
            aligned_times.append(audio_times[audio_time_index])
            aligned_pitches.append(audio_pitches[audio_time_index])
            audio_time_index += 1
        aligned_times.append(midi_time)
        aligned_pitches.append(midi_pitch)
    return aligned_pitches, aligned_times

# def measure_time(func, *args):
#     start_time = time.time()
#     result = func(*args)
#     elapsed_time = time.time() - start_time
#     return result, elapsed_time

def main():

    audio_filename = "datasets/recordings/C Scale 1.wav"
    # midi_filename = "datasets/generated_scales/C Major/C Scale .mid"

    audio_loader = ess.MonoLoader(filename=audio_filename)
    audio = audio_loader()

    eq_loudness = ess.EqualLoudness()
    cleaned_audio = eq_loudness(audio)
    melodia = ess.PitchMelodia(frameSize=framesize, hopSize=hopsize)
    pitches_melodia = compute_pitches_no_frames(cleaned_audio, melodia)

    print("*****PitchMelodia*****")
    # print(type(pitches_melodia)) #numpy.ndarray
    # print(f'    - pitches: {pitches_melodia}')
    print(f'    - length {len(pitches_melodia)}')

    predominant_melodia = ess.PredominantPitchMelodia(frameSize=framesize, hopSize=hopsize)
    pitches_predominant_melodia = compute_pitches_no_frames(audio, predominant_melodia)

    print("*****PredominantPitchMelodia*****")
    # print(type(pitches_predominant_melodia)) #numpy.ndarray
    # print(f'    - pitches: {pitches_predominant_melodia}')
    print(f'    - length {len(pitches_predominant_melodia)}')

    pitches_yin = []
    pitches_yinfft = []
    pitches_yinprobabilities = []

    for frame in ess.FrameGenerator(audio, frameSize=framesize, hopSize=hopsize, startFromZero=True):
        pitches_yin.append(pitchyin(frame))
        pitches_yinfft.append(pitchyinfft(frame))
        pitches_yinprobabilities.append(pitchyinprobabilities(frame))

    print("*****PitchYin*****")
    # print(type(pitches_yin)) #list
    # print(f'    - pitches: {pitches_yin}')
    print(f'    - length: {len(pitches_yin)}')

    print("*****PitchYinFFT*****")
    # print(type(pitches_yinfft)) #list
    # print(f'    - pitches: {pitches_yinfft}')
    print(f'    - length: {len(pitches_yinfft)}')

    pitch_yin_probabilistic = ess.PitchYinProbabilistic(frameSize=framesize, hopSize=hopsize)
    pitches_yinprobabilistic, _ = pitch_yin_probabilistic(audio)

    print("*****PitchYinProbabilistic*****")
    # print(type(pitches_yinprobabilistic))  #numpy.ndarray
    # print(f'    - pitches: {pitches_yinprobabilistic}')
    print(f'    - length: {len(pitches_yinprobabilistic)}')

    print("*****PitchYinProbabilities*****")
    # print(type(pitches_yinprobabilities))  #list
    # print(f'    - pitches: {pitches_yinprobabilities}')
    print(f'    - length: {len(pitches_yinprobabilities)}')



    # midi_notes = load_midi(midi_filename)
    # aligned_pitches, aligned_times = align_pitches(audio_pitches, np.arange(len(audio_pitches)), midi_notes)


    # accuracy = mir_eval.melody.evaluate(np.asarray(midi_notes)[:,0], np.asarray(midi_notes)[:,1], np.asarray(aligned_pitches), np.asarray(aligned_times))
    # print(f" Melodia Accuracy: {accuracy}")



if __name__ == "__main__":
    main()