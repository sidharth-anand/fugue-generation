import os
import sys
import glob
import pathlib
import functools
import multiprocessing

import muspy

import music21

import numpy as np

def inverse_dict(d):
    """Return the inverse dictionary."""
    return {v: k for k, v in d.items()}

RESOLUTION = 24
MAX_BEAT = 1024
MAX_DURATION = 384  # Remember to modify known durations as well!

# Dimensions
# (NOTE: "type" must be the first dimension!)
# (NOTE: Remember to modify N_TOKENS as well!)
DIMENSIONS = ["type", "beat", "position", "pitch", "duration", "instrument"]
assert DIMENSIONS[0] == "type"

# Type
TYPE_CODE_MAP = {
    "start-of-song": 0,
    "instrument": 1,
    "start-of-notes": 2,
    "note": 3,
    "end-of-song": 4,
}
CODE_TYPE_MAP = inverse_dict(TYPE_CODE_MAP)

# Beat
BEAT_CODE_MAP = {i: i + 1 for i in range(MAX_BEAT + 1)}
BEAT_CODE_MAP[None] = 0
CODE_BEAT_MAP = inverse_dict(BEAT_CODE_MAP)

# Position
POSITION_CODE_MAP = {i: i + 1 for i in range(RESOLUTION)}
POSITION_CODE_MAP[None] = 0
CODE_POSITION_MAP = inverse_dict(POSITION_CODE_MAP)

# Pitch
PITCH_CODE_MAP = {i: i + 1 for i in range(128)}
PITCH_CODE_MAP[None] = 0
CODE_PITCH_MAP = inverse_dict(PITCH_CODE_MAP)

# Duration
KNOWN_DURATIONS = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    15,
    16,
    18,
    20,
    21,
    23,
    24,
    30,
    36,
    40,
    42,
    48,
    60,
    72,
    84,
    96,
    120,
    144,
    168,
    192,
    384,
]
DURATION_CODE_MAP = {
    i: int(np.argmin(np.abs(np.array(KNOWN_DURATIONS) - i))) + 1
    for i in range(MAX_DURATION + 1)
}
DURATION_CODE_MAP[None] = 0
CODE_DURATION_MAP = {
    i + 1: duration for i, duration in enumerate(KNOWN_DURATIONS)
}

MAX_VOICES = 4

N_TOKENS = [
    max(TYPE_CODE_MAP.values()) + 1,
    max(BEAT_CODE_MAP.values()) + 1,
    max(POSITION_CODE_MAP.values()) + 1,
    max(PITCH_CODE_MAP.values()) + 1,
    max(DURATION_CODE_MAP.values()) + 1,
    MAX_VOICES,
]

CHUNK_SIZE = 8
MIN_NOTES = 20

def get_encoding():
    """Return the encoding configurations."""
    return {
        "resolution": RESOLUTION,
        "max_beat": MAX_BEAT,
        "max_duration": MAX_DURATION,
        "dimensions": DIMENSIONS,
        "n_tokens": N_TOKENS,
        "type_code_map": TYPE_CODE_MAP,
        "beat_code_map": BEAT_CODE_MAP,
        "position_code_map": POSITION_CODE_MAP,
        "pitch_code_map": PITCH_CODE_MAP,
        "duration_code_map": DURATION_CODE_MAP,
        "code_type_map": CODE_TYPE_MAP,
        "code_beat_map": CODE_BEAT_MAP,
        "code_position_map": CODE_POSITION_MAP,
        "code_pitch_map": CODE_PITCH_MAP,
        "code_duration_map": CODE_DURATION_MAP,
        "max_voices": MAX_VOICES,
    }

def chunk_music(file):
    '''Get a file and return a list of muspy object chunked into bars'''

    music = muspy.to_music21(muspy.read_midi(file))

    #No metadata
    parts = music[1:]   

    # Chunk the music   
    chunks = []

    [part.makeMeasures(inPlace=True) for part in parts]

    offset = parts[0].getElementsByClass('Measure')[1].offset

    max_measure = max([len(part) for part in parts]) // CHUNK_SIZE * CHUNK_SIZE  # drop extra measures at the end
    chunk_indices = [(i, i + CHUNK_SIZE) for i in range(max_measure)]

    for start, end in chunk_indices:
        skip = False

        _chunks = [music21.stream.Part(part[start:end]) for part in parts]

        for _chunk in _chunks:
            if len(_chunk.flatten().getElementsByClass('Note')) < MIN_NOTES:
                skip = True
                break

            for i, measure in enumerate(_chunk):
                measure.offset = offset * i

        if skip:
            continue
        
        chunks.append(music21.stream.Score([music[0]] + _chunks))

    chunks = [muspy.from_music21(chunk, resolution=RESOLUTION) for chunk in chunks]

    return chunks


def extract_notes(music, resolution):
    """Return a MusPy music object as a note sequence.

    Each row of the output is a note specified as follows.

        (beat, position, pitch, duration, program)

    """
    # Extract notes
    notes = []
    for i, track in enumerate(music):
        for note in track:
            beat, position = divmod(note.time, resolution)
            notes.append(
                (beat, position, note.pitch, note.duration, i)
            )

    # Deduplicate and sort the notes
    notes = sorted(set(notes), key=lambda x: x[-1])

    return np.array(notes)

def encode_notes(notes, encoding):
    """Encode a note sequence into a sequence of codes.

    Each row of the input is a note specified as follows.

        (beat, position, pitch, duration, program)

    Each row of the output is encoded as follows.

        (event_type, beat, position, pitch, duration, instrument)

    """
    max_beat = encoding["max_beat"]
    max_duration = encoding["max_duration"]

    # Get maps
    type_code_map = encoding["type_code_map"]
    beat_code_map = encoding["beat_code_map"]
    position_code_map = encoding["position_code_map"]
    pitch_code_map = encoding["pitch_code_map"]
    duration_code_map = encoding["duration_code_map"]

    beat_dim = encoding["dimensions"].index("beat")
    position_dim = encoding["dimensions"].index("position")
    pitch_dim = encoding["dimensions"].index("pitch")
    duration_dim = encoding["dimensions"].index("duration")
    instrument_dim = encoding["dimensions"].index("instrument")

    codes = [(type_code_map["start-of-song"], 0, 0, 0, 0, 0)]
    instruments = set(note[-1] for note in notes)

    instrument_codes = []
    for instrument in instruments:
        if instrument is None:
            continue
        row = [type_code_map["instrument"], 0, 0, 0, 0, 0]
        row[instrument_dim] = instrument
        instrument_codes.append(row)

    # Sort the instruments and append them to the code sequence
    instrument_codes.sort()
    codes.extend(instrument_codes)

    codes.append((type_code_map["start-of-notes"], 0, 0, 0, 0, 0))

    for beat, position, pitch, duration, voice in notes:
        # Skip if max_beat has reached
        if beat > max_beat:
            continue
        # Skip unknown instruments

        # Encode the note
        row = [type_code_map["note"], 0, 0, 0, 0, 0]

        row[beat_dim] = beat_code_map[beat]
        row[position_dim] = position_code_map[position]
        row[pitch_dim] = pitch_code_map[pitch]
        row[duration_dim] = duration_code_map[min(duration, max_duration)]
        row[instrument_dim] = voice

        codes.append(row)

    codes.append((type_code_map["end-of-song"], 0, 0, 0, 0, 0))

    return np.array(codes)

def decode_notes(codes, encoding):
    """Decode codes into a note sequence.

    Each row of the input is encoded as follows.

        (event_type, beat, position, pitch, duration, instrument)

    """
    # Get variables and maps
    code_type_map = encoding["code_type_map"]
    code_beat_map = encoding["code_beat_map"]
    code_position_map = encoding["code_position_map"]
    code_pitch_map = encoding["code_pitch_map"]
    code_duration_map = encoding["code_duration_map"]

    # Get the dimension indices
    beat_dim = encoding["dimensions"].index("beat")
    position_dim = encoding["dimensions"].index("position")
    pitch_dim = encoding["dimensions"].index("pitch")
    duration_dim = encoding["dimensions"].index("duration")
    instrument_dim = encoding["dimensions"].index("instrument")

    # Decode the codes into a sequence of notes
    notes = []
    for row in codes:
        event_type = code_type_map[int(row[0])]
        if event_type in ("start-of-song", "instrument", "start-of-notes"):
            continue
        elif event_type == "end-of-song":
            break
        elif event_type == "note":
            beat = code_beat_map[int(row[beat_dim])]
            position = code_position_map[int(row[position_dim])]
            pitch = code_pitch_map[int(row[pitch_dim])]
            duration = code_duration_map[int(row[duration_dim])]
            instrument = int(row[instrument_dim])
            notes.append((beat, position, pitch, duration, instrument))
        else:
            raise ValueError("Unknown event type.")

    return np.array(notes)


def reconstruct(notes, resolution):
    """Reconstruct a note sequence to a MusPy Music object."""
    # Construct the MusPy Music object
    music = muspy.Music(resolution=resolution)

    # Append the tracks
    voices = np.max(notes[:, -1]) + 1   
    for _ in range(voices):
        music.tracks.append(muspy.Track())

    # Append the notes
    for beat, position, pitch, duration, voice in notes:
        time = beat * resolution + position
        music[voice].notes.append(muspy.Note(time, pitch, duration))

    return music

def chunk_and_encode(file, target):
    name = pathlib.Path(file).stem

    chunks = chunk_music(file)
    encoding = get_encoding()

    for i, _chunk in enumerate(chunks):
        notes = extract_notes(_chunk, encoding["resolution"])
        encoded = encode_notes(notes, encoding)

        np.save(os.path.join(target, f'{name}_c_{i}'), encoded)

if __name__ == '__main__':
    
    root = sys.argv[1]

    src = os.path.join(root, 'midi')
    target = os.path.join(root, 'chunked')

    if not os.path.exists(target):
        os.makedirs(target)

    files = glob.glob(os.path.join(src, '*.mid'))

    print(f'Processing {len(files)} files. Storing results in {target}...')

    with multiprocessing.Pool(os.cpu_count()) as pool:
        results = pool.map(functools.partial(chunk_and_encode, target=target), files)