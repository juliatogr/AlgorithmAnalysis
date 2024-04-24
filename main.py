import essentia.standard as ess
import mir_eval
import numpy as np
import time
import mido


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

def measure_time(func, *args):
    start_time = time.time()
    result = func(*args)
    elapsed_time = time.time() - start_time
    return result, elapsed_time

def compute_pitch(audio, algorithm='melodia'):
    if algorithm == 'melodia':
        pitch_extractor = ess.PitchMelodia()
    elif algorithm == 'yin':
        pitch_extractor = ess.PitchYinFFT()
    else:
        raise ValueError("Invalid algorithm name")

    pitches, pitch_confidence = pitch_extractor(audio)
    return pitches, pitch_confidence

def main():

    # pitch_extractors = [pitch_melodia, pitch_yin, pitch_yin_fft, predominant_pitch_melodia, spectral_peaks]
    #pitch_extractors = [pitch_melodia]

    audio_filename = "datasets/recordings/C Major recorded.wav"
    midi_filename = "datasets/recordings/C Scale .mid"
    audio_loader = ess.MonoLoader(filename=audio_filename)
    audio = audio_loader()

    # Compute pitch using PitchMelodia
    melodia_pitches, _ = measure_time(compute_pitch, audio, 'melodia')

    # Compute pitch using PitchYin
    yin_pitches, _ = measure_time(compute_pitch, audio, 'yin')

    # Compute pitch using PitchMelodia
    melodia_extractor = ess.PitchMelodia()
    audio_pitches, audio_confidence = melodia_extractor(audio)

    # Load MIDI and align pitches
    midi_notes = load_midi(midi_filename)
    aligned_pitches, aligned_times = align_pitches(audio_pitches, np.arange(len(audio_pitches)), midi_notes)

    # Evaluate accuracy
    accuracy = mir_eval.melody.evaluate(np.asarray(midi_notes)[:,0], np.asarray(midi_notes)[:,1], np.asarray(aligned_pitches), np.asarray(aligned_times))
    print(f" Melodia Accuracy: {accuracy}")

    yin_extractor = ess.PitchYin()
    audio_pitches, audio_confidence = yin_extractor(audio)

    # Load MIDI and align pitches
    midi_notes = load_midi(midi_filename)
    aligned_pitches, aligned_times = align_pitches(audio_pitches, np.arange(len(audio_pitches)), midi_notes)

    # Evaluate accuracy
    accuracy = mir_eval.melody.evaluate(np.asarray(midi_notes)[:, 0], np.asarray(midi_notes)[:, 1],
                                        np.asarray(aligned_pitches), np.asarray(aligned_times))
    print(f" Yin Accuracy: {accuracy}")

if __name__ == "__main__":
    main()