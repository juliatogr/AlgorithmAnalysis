import essentia.standard as es


def extract(audio):
    """
    Extracts melody frequency in Hz from audio using Yin algorithm.
    :param audio: audio signal
    :return: frequencies of the melody
    """

    print('PitchYinFFT')
    algo = es.PitchYinFFT()
    return algo(audio)
