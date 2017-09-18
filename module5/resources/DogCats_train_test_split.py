#!/usr/bin/python3

"""
Original Dataset:
DogsCats/
    train/
        cat.001.jpg
        cat.002.jpg
        dog.001.jpg
        dog.002.jpg
    test1/

This script adds:
DogsCats/
    TransferLearning/
        train/
            dogs/
                001.jpg => ../../../train/dog.001.jpg
                002.jpg => ../../../train/dog.002.jpg
                ...
            cats/
                001.jpg => ../../../train/cat.001.jpg
                002.jpg => ../../../train/cat.002.jpg
                ...
        validation/
            dogs/
            cats/
    train/
    test1/
    
To undo whatever this script did:

$ cd "$DATASET" # which contains same value as `DATASET` below
$ rm -rf TransferLearning/
"""

import os, sys, re, random
import numpy as np

# Dataset path
DATASET = '/dsa/data/all_datasets/DogsCats'
assert os.path.exists(DATASET)

os.chdir(DATASET)
metadata = {'dog': [], 'cat': []}
for fname in os.listdir('train'):
    match = re.match(r'^(\w+)\.(\d+\.jpg)$', fname)
    if match:
        metadata[match.group(1)].append((
            os.path.join('../../../train', match.group(0)),
            match.group(2)
        ))

random.shuffle(metadata['dog'])
random.shuffle(metadata['cat'])

print('Found', {k:len(v) for k,v in metadata.items()})
os.system('rm -rf TransferLearning && mkdir -pv TransferLearning/{train,validation}/{dogs,cats}')

def link(src, dst):
    return os.symlink(src, dst)
    try:
        os.symlink(src, dst)
        print(src, '<=', dst)
    except FileExistsError:
        print(dst, 'exists')

for target, split in [('TransferLearning/train/dogs', metadata['dog'][400:]),
                        ('TransferLearning/train/cats', metadata['cat'][400:]),
                        ('TransferLearning/validation/dogs', metadata['dog'][:400]),
                        ('TransferLearning/validation/cats', metadata['cat'][:400])]:
    [link(source, os.path.join(target, link_name)) for source, link_name in split]
    print(len(split), 'symlinks created for', target)

    