import essentia.standard as es


def extract(audio):

    print('PredominantPitchMelodia')
    algo = es.PredominantPitchMelodia()
    pitch, confidence = algo(audio)
    return pitch(audio)

