import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import os


image_feature_description = {
    'label': tf.io.FixedLenFeature([], tf.int64),
    'image_raw': tf.io.FixedLenFeature([], tf.string),
}


def _parse_image_function(example_proto):
    # Parse the input tf.train.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, image_feature_description)


def parse_img(feature):
    # img = tf.io.read_file(feature['image_raw'])
    img = feature['image_raw']
    return tf.cast(tf.image.decode_jpeg(img), tf.float32)


def map_img_color(img):
    img_unstacked = tf.unstack(img, axis=-1)
    img = tf.stack([img_unstacked[2], img_unstacked[1], img_unstacked[0]], axis=-1)
    img = img / 255.0
    return img


def parse_img_label_3(example, img_size=224):
    parsed = _parse_image_function(example)
    img = parse_img(parsed)
    img = tf.ensure_shape(img, (img_size,img_size,3))
    img = map_img_color(img)
    label = parsed['label']

    return img, label

def data_aug(x, y):
  img = x
  img = tf.image.random_flip_left_right(img)
  #img = tf.image.per_image_standardization(img)
  return img, y

def img_rotation(x,y):
  angle = np.random.uniform(low=-np.pi/18, high=np.pi/18)
  x = tfa.image.rotate(x, angle, fill_mode='nearest')
  return x, y

def data_formatting(x,y):
  return (x,y),y


random_rotate = tf.keras.Sequential([tf.keras.layers.RandomRotation(1/36)])


def dataset_from_shards(tfrecords_dir, tfrec_name, batch_size=64, rotation=False, arcface_format=False):
    """

    :param tfrecords_dir: path to directory where the tfrecord shards are
    :param tfrec_name: common name of the tfrecord shards
    :param batch_size: number of samples per batch
    :return: a tf.data.dataset object, with pairwise shuffling applied
    """

    file_dir = os.path.join(tfrecords_dir, '%s_*' % tfrec_name)
    files = tf.io.matching_files(file_dir)
    shard_list = tf.data.Dataset.from_tensor_slices(files)

    # Load and parse training split tfrecords
    dataset = shard_list.interleave(lambda x: tf.data.TFRecordDataset(x),
                                    cycle_length=4,
                                    block_length=2,
                                    num_parallel_calls=tf.data.AUTOTUNE,
                                    deterministic=False)
    # NOTE: cycle length is number of input elements to process simultaneously,
    #       block_length is num of consecutive elements (2)


    # Pairwise shuffling of dataset
    dataset = dataset.batch(2)
    #dataset = dataset.shuffle(buffer_size=8192)
    dataset = dataset.shuffle(buffer_size=16384)
    dataset = dataset.unbatch()

    dataset = dataset.map(parse_img_label_3)
    dataset = dataset.map(data_aug)
    if rotation:
        dataset = dataset.map(img_rotation)
    if arcface_format:
        dataset = dataset.map(data_formatting)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset