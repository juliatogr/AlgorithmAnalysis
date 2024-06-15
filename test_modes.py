import essentia
import essentia.standard as esstd
import essentia.streaming as esstr
import matplotlib.pyplot as plt

hopsize = 128
framesize = 8192
sample_rate = 44100
num_hps = 5

def main():
    audio_filename = './datasets/Traditional Flute/audio/allemande_fifth_fragment_preston.wav'

    spectrum = esstd.Spectrum()
    w = esstd.Windowing(type='hann')
    loader = esstd.MonoLoader(filename=audio_filename)
    audio = loader()
    frame_generator = esstd.FrameGenerator(audio, frameSize=framesize, hopSize=hopsize)
    hps = esstd.PitchHPS(frameSize=framesize, numHarmonics=num_hps)

    pitches = []
    for frame in frame_generator:
        spec = spectrum(w(frame))
        pitch, _ = hps(spec)
        pitches.append(pitch)

    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    ax.plot(pitches, label='HPS')
    ax.set_xlabel('Frames')
    ax.set_ylabel('Estimated pitch')
    ax.set_xlim(1000, 2000)
    ax.set_ylim(0, 1000)
    ax.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig("./images/standard.png")

    # Streaming
    loader = esstr.MonoLoader(filename=audio_filename)
    frameCutter = esstr.FrameCutter(frameSize=framesize, hopSize=hopsize)
    spectrum = esstr.Spectrum()
    w = esstr.Windowing(type='hann')

    hps = esstr.PitchHPS()
    loader.audio >> frameCutter.signal
    frameCutter.frame >> w.frame >> spectrum.frame
    spectrum.spectrum >> hps.spectrum

    pool = essentia.Pool()

    hps.pitch >> (pool, 'pitches_hps')
    hps.pitchConfidence >> (pool, 'pitchConfidence_hps')

    essentia.run(loader)

    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    ax.plot(pool['pitches_hps'], label='HPS')
    ax.set_xlabel('Frames')
    ax.set_ylabel('Estimated pitch')
    ax.set_xlim(1000, 2000)
    ax.set_ylim(0, 1000)
    ax.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig("./images/streaming.png")



if __name__ == "__main__":
    main()
