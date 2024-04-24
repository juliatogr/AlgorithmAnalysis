import essentia.standard as es


def extract(audio):
    print('SpectralPeaks')
    spectrum = es.SpectralPeaks()

    return spectrum(audio)
