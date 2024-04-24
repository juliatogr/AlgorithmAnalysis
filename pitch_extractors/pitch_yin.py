import essentia.standard as es


def extract(audio):

    print('PitchYin')
    algo = es.PitchYin()
    return algo(audio)
