import mido

def get_midi_info(gt_path):
    """
    Get the midi info from the ground truth file
    :param gt_path:
    :return: frequencies, notes, onsets, durations
    """

    midi = mido.MidiFile(gt_path)

    general_info_track = midi.tracks[0]
    print('general_info_track')
    bpm = 120

    for msg in general_info_track:
        print(msg)
        # if msg.type == 'set_tempo':
        #     print('bpm='+str(mido.tempo2bpm(msg.tempo)))

    print('messages')
    for msg in midi.tracks[1]:
        print(msg)
        # if msg.type == 'note_on':
        #     freq = 440 * (2**((msg.note - 69)/12))
        #     print(freq)

    return midi

if __name__ == "__main__":
    midi_filename = "datasets/songs/Canon in D - Pachebel/_Canon_de_Pachelbel_Flauta_dulce_soprano.mid"
    print("path" + midi_filename)
    get_midi_info(midi_filename)
