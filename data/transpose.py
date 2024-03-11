import os
import sys
import glob
import functools
import multiprocessing

import numpy as np

from data.encode import get_encoding

encoding = get_encoding()

TRANSPOSITIONS = 5
TRANSPOSE_RANGE = 11

def transpose_and_write(file, target):
    data = np.load(file)

    shift_mask = np.where(data[:, encoding['dimensions'].index('type')] == encoding['type_code_map']['note'], 1, 0 )

    for _ in range(TRANSPOSITIONS):
        shift = np.random.randint(-TRANSPOSE_RANGE, TRANSPOSE_RANGE + 1)
        
        modifier = np.zeros_like(data)
        modifier[:, encoding['dimensions'].index('pitch')] = shift_mask * shift

        np.save(os.path.join(target, f'{os.path.basename(file)}_t_{shift}.npy'), data + modifier)

    np.save(os.path.join(target, f'{os.path.basename(file)}_t_0.npy'), data)

if __name__ == '__main__':
    root = sys.argv[1]

    src = os.path.join(root, 'chunked')
    target = os.path.join(root, 'transposed')

    if not os.path.exists(target):
        os.makedirs(target)

    files = glob.glob(os.path.join(src, '*.npy'))

    print(f'Found {len(files)} files to transpose. Transposing each {TRANSPOSITIONS} times and writing to {target}...')

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        pool.map(functools.partial(transpose_and_write, target=target), files)