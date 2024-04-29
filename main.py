import essentia.standard as ess
import numpy as np

spectrum = ess.Spectrum()
w = ess.Windowing(type='hann')
framesize = 2048
hopsize = 128
sample_rate = 44100


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


def main():

    audio_filename = "datasets/recordings/C Scale 1.wav"
    midi_filename = "datasets/generated_scales/C Major/C Scale .mid"

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


if __name__ == "__main__":
    main()