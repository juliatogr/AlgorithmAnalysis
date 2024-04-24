import essentia.standard as es


def extract(audio):
    print('FFT')
    spectrum = es.FFT()

    return spectrum(audio)
