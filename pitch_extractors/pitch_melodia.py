import essentia.standard as es

def extract(audio):
    print('PitchMelodia')
    algo = es.PitchMelodia()
    pitch, confidence = algo(audio)
    return pitch

