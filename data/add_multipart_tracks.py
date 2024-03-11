import os
import glob
import pathlib
import itertools
import multiprocessing

import mido
import tqdm

import numpy as np

TWOPART_DIR = 'data/baroque_small_trainset_twopart'
MULTIPART_DIR = 'data/baroque_small_trainset_multipart'

SOURCE_DIR = os.path.join(TWOPART_DIR, 'midi')
TARGET_DIR = os.path.join(MULTIPART_DIR, 'midi')

MIN_TRACKS = 2
MAX_TRACKS = 4

def is_metadata_track(track):
    for i, msg in enumerate(track):
        if msg.type == 'note_on' and msg.velocity > 0:
            return False
    return True

def get_avg_pitch_height(track):
    pitches = []
    for msg in track:
        if msg.type == 'note_on' and msg.velocity > 0:
            pitches.append(msg.note)
    return np.mean(pitches)

def write_multipart_midi(name):
    parts = glob.glob(os.path.join(SOURCE_DIR, f'{name}*.mid'))

    meta_track = None
    ticks_per_beat = None

    tracks = {}

    for part in parts:
        mid = mido.MidiFile(part)

        if ticks_per_beat is None:
            ticks_per_beat = mid.ticks_per_beat
        else:
            if ticks_per_beat != mid.ticks_per_beat:
                print(f'Inconsistent ticks per beat: {ticks_per_beat} vs {mid.ticks_per_beat}')

        indices = pathlib.Path(part).stem.split('_')[-2:]

        for i, track in enumerate(mid.tracks):
            if i == 0 :
                if meta_track is None:
                    meta_track = track
                
                continue
            
            if indices[i - 1] not in tracks:
                tracks[indices[i - 1]] = track

    for track_count in range(MIN_TRACKS, MAX_TRACKS + 1):
        if len(tracks) < track_count:
            continue

        for selected_tracks in itertools.combinations(tracks.keys(), track_count):
            mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
            mid.tracks.append(meta_track)

            for track in selected_tracks:
                mid.tracks.append(tracks[track])

            mid.save(os.path.join(TARGET_DIR, f'{name}_{"_".join(selected_tracks)}.mid'))

def main():
    if not os.path.exists(MULTIPART_DIR):
        os.makedirs(MULTIPART_DIR)

    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)

    sources = os.listdir(SOURCE_DIR)

    process = []

    for source in sources:
        original = '_'.join(source.split('_')[:-2])

        if not original in process:
            process.append(original)

    print(f'Starting to process {len(process)} files...', )

    with multiprocessing.Pool(os.cpu_count()) as pool:
        pool.map(write_multipart_midi, process)

if __name__ == '__main__':
    main()