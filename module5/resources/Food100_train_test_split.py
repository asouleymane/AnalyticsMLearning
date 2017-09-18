#!/usr/bin/python3

"""
Original Dataset:
UECFOOD100/
    1/
        1.jpg
        2.jpg
        3.jpg
        ...
    2/
    ...

This script adds:
UECFOOD100/
    TransferLearning/
        train/
            1/
                %.jpg => ../../../%.jpg
                ...
            2/
            ...
        validation/
            1/
            2/
            ...
            
To undo whatever this script did:

$ cd "$DATASET" # which points to same path as `DATASET` below
$ rm -rf TransferLearning/
"""

import os, sys, re, random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from skimage.io import imread

# Dataset path
DATASET = lambda fname = '': os.path.join('/dsa/data/all_datasets/UECFOOD100', fname)
assert os.path.exists(DATASET())

os.chdir(DATASET())
categories = {str(i):food for i, food in pd.read_csv(DATASET('category.txt'), sep='\t').values}
metadata = {food:list() for food in categories.values()}
for path, dirs, files in os.walk('.'):
    category_id = os.path.basename(path)
    if re.match('^\d+$', category_id):
        metadata[categories[category_id]].extend(
            (os.path.join(category_id, fname), fname)
                for fname in files if os.path.splitext(fname)[1] in ['.jpg'])

for category in metadata:
    random.shuffle(metadata[category])

print('Found', {k:len(v) for k,v in metadata.items()})
print('    Min', np.min([len(v) for v in metadata.values()]))
print('    Max', np.max([len(v) for v in metadata.values()]))

# This command may be too long...
# os.system('rm -rf TransferLearning && mkdir -p TransferLearning/{train,validation}/{%s}'%','.join(map(str,categories.keys())))

os.system('rm -rf TransferLearning && mkdir -p TransferLearning/{train,validation}')
os.system('cd TransferLearning/train && mkdir '+'\x20'.join(map(str,categories.keys())))
os.system('cd TransferLearning/validation && mkdir '+'\x20'.join(map(str,categories.keys())))

# Don't know why this could fail... try `rm -rf TransferLearning` manually upon FileExistsError etc

def link(src, dst):
    return os.symlink(src, dst)
    try:
        os.symlink(src, dst)
        print(src, '<=', dst)
    except FileExistsError:
        print(dst, 'exists')

import warnings
warnings.filterwarnings('error') # Take warnings as errors

bad_images = []
def test_open(im_name):
    """ Test if image can be opened with packages installed on system, which probably depends on PIL,
        otherwise it would interrupt during training anyway. """
    global fail_counter
    try:
        imread(im_name)
    except (IOError, ZeroDivisionError, UserWarning) as e:
        bad_images.append(im_name) 
        return False
    return True
    
total_train = 0
total_test = 0
for i, category in categories.items():
    data_train, data_test = train_test_split(metadata[category], test_size = .2)
    target_train = lambda s: os.path.join(os.path.join('TransferLearning/train', str(i)), s)
    target_test = lambda s: os.path.join(os.path.join('TransferLearning/validation', str(i)), s)
    link_source =  lambda s: os.path.join('../../../', s)
    num_train = len([link(link_source(source), target_train(link_name))
        for source, link_name in data_train if test_open(source)])
    num_test = len([link(link_source(source), target_test(link_name))
        for source, link_name in data_test if test_open(source)])
    print(num_train+num_test, 'symlinks created for', category)
    total_train += num_train
    total_test += num_test
    
print('    train/test ratio:', total_train, ':', total_test)
print("Can't open", len(bad_images), "images. They are excluded:", bad_images)