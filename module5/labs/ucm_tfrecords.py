#!/usr/bin/env python3

import os, sys
import random, math
import tensorflow as tf
import tensorflow.contrib.slim as slim

# borrowed from https://github.com/tensorflow/models/blob/master/slim/datasets/dataset_utils.py

def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def image_to_tfexample(image_data, image_format, height, width, class_id):
    return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': bytes_feature(image_data),
      'image/format': bytes_feature(image_format),
      'image/class/label': int64_feature(class_id),
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
    }))

def write_label_file(labels_to_class_names, dataset_dir, filename='labels.txt'):
    """Writes a file with the list of class names.
    Args:
        labels_to_class_names: A map of (integer) labels to class names.
        dataset_dir: The directory in which the labels file should be written.
        filename: The filename where the class names are written.
    """
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'w') as f:
        for label in labels_to_class_names:
            class_name = labels_to_class_names[label]
            f.write('%d:%s\n' % (label, class_name))

_FILE_PATTERN = 'UCMerced_%s_*.tfrecord'
SPLITS_TO_SIZES = {'train': 1680, 'validation': 420}
_NUM_CLASSES = 21
_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'label': 'A single integer between 0 and 21',
}
_NUM_SHARDS = 5

class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_image(self._decode_jpeg_data, channels=3)

    def read_image_dims(self, sess, image_data):
        image = self.decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_jpeg(self, sess, image_data):
        image = sess.run(self._decode_jpeg, feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _get_filenames_and_classes(dataset_dir):
    """Returns a list of filenames and inferred class names.

    Args:
        dataset_dir: A directory containing a set of subdirectories representing
            class names. Each subdirectory should contain PNG or JPG encoded images.

    Returns:
        A list of image file paths, relative to `dataset_dir` and the list of
        subdirectories, representing class names.
    """
    image_root = os.path.join(dataset_dir)
    directories = []
    class_names = []
    for filename in os.listdir(image_root):
        path = os.path.join(image_root, filename)
        if os.path.isdir(path):
            directories.append(path)
            class_names.append(filename)

    photo_filenames = []
    for directory in directories:
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            photo_filenames.append(path)

    return photo_filenames, sorted(class_names)


def _get_dataset_filename(dataset_dir, split_name, shard_id):
    output_filename = 'UCMerced_%s_%05d-of-%05d.tfrecord' % (
            split_name, shard_id, _NUM_SHARDS)
    return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name, filenames, class_names_to_ids, dataset_dir, image_dir):
    """Converts the given filenames to a TFRecord dataset.

    Args:
        split_name: The name of the dataset, either 'train' or 'validation'.
        filenames: A list of absolute paths to png or jpg images.
        class_names_to_ids: A dictionary from class names (strings) to ids
            (integers).
        dataset_dir: The directory where the converted datasets are stored.
    """
    assert split_name in ['train', 'validation']

    num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

    with tf.Graph().as_default():
        image_reader = ImageReader()

        with tf.Session('') as sess:

            for shard_id in range(_NUM_SHARDS):
                output_filename = _get_dataset_filename(
                        dataset_dir, split_name, shard_id)

                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
                    for i in range(start_ndx, end_ndx):
                        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                                i+1, len(filenames), shard_id))
                        sys.stdout.flush()

                        # Read the filename:
                        image_data = tf.gfile.FastGFile(image_dir + '/' + filenames[i], 'rb').read()
                        height, width = image_reader.read_image_dims(sess, image_data)

                        class_name = os.path.basename(os.path.dirname(filenames[i]))
                        class_id = class_names_to_ids[class_name]

                        example = image_to_tfexample( # dataset_utils
                                image_data, b'png', height, width, class_id)
                        tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()


def _clean_up_temporary_files(dataset_dir):
    """Removes temporary files used to create the dataset.

    Args:
        dataset_dir: The directory where the temporary files are stored.
    """
    return

def _dataset_exists(dataset_dir):
    for split_name in ['train', 'validation']:
        for shard_id in range(_NUM_SHARDS):
            output_filename = _get_dataset_filename(
                    dataset_dir, split_name, shard_id)
            if not tf.gfile.Exists(output_filename):
                return False
    return True

def read_fivefold_list(dataset_dir, foldList):
        """Reads the file list of the fivefold cross validation and
        uses it to make a list"""
        
        foldList_filename = os.path.join(dataset_dir, foldList)
        with tf.gfile.Open(foldList_filename, 'r') as f:
                lines = f.read()
        lines = lines.split('\n')
        lines = lines[:-1]
        
        return lines

def create_TF_RecordsForFold(folds_dir, image_dir, dataset_dir, fold):
    """Runs the download and conversion operation.

    Args:
        dataset_dir: The dataset directory where the dataset is stored.
    """
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    if _dataset_exists(dataset_dir):
        print('Dataset files already exist. Exiting without re-creating them.')
        return

    photo_filenames, class_names = _get_filenames_and_classes(image_dir)
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))


    # This function uses file lists created via bash scripts.
    training_filenames = read_fivefold_list(folds_dir, 'fivefold_' + fold + '_train.txt')
    random.shuffle(training_filenames)
    validation_filenames = read_fivefold_list(folds_dir, 'fivefold_' + fold + '_test.txt')

    # First, convert the training and validation sets.
    _convert_dataset('train', training_filenames, class_names_to_ids,
                             dataset_dir, image_dir )
    _convert_dataset('validation', validation_filenames, class_names_to_ids,
                             dataset_dir, image_dir )

    # Finally, write the labels file:
    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    write_label_file(labels_to_class_names, dataset_dir)

    #_clean_up_temporary_files(dataset_dir)
    print('\nFinished converting the UCMerced dataset!')
    
image_dir = '/dsa/data/all_datasets/ucm'
folds_dir = '/dsa/data/all_datasets/ucm_Fivefolds'

# Local TF Record data location
dataset_dir = './data'

# Folds go from A - E
fold = 'A'

create_TF_RecordsForFold(folds_dir, image_dir, dataset_dir, fold)