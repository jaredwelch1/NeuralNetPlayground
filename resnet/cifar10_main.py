""" Run Resnet model with CIFAR10 dataset """

import os
import sys

import tensorflow as tf

import resnet

_HEIGHT = 32
_WIDTH = 32
_NUM_CHANNELS = 3
_DEFAULT_IMAGE_BYTES = _HEIGHT * _WIDTH * _NUM_CHANNELS
# record is image and a one-byte label

_RECORD_BYTES = _DEFAULT_IMAGE_BYTES + 1
_NUM_CLASSES = 10
_NUM_DATA_FILES = 5

_NUM_IMAGES = {
    'train' : 50000,
    'validation' : 10000,
}


###############################################################################
# Data processing functions
###############################################################################
def get_filnames(is_training, data_dir):
    """ Returns list of file names from data_dir """
    data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')

    assert os.path.exists(data_dir), (
        'Run cifar10_download_extract.py to get the data %s', data_dir)

    if is_training:
        return [
            os.path.join(data_dir, 'data_batch_%d.bin' % i)
            for i in range(1, _NUM_DATA_FILES + 1)
        ]
    return [os.path.join(data_dir, 'test_batch.bin')]

def parse_record(raw_record, is_training):
    """ Parse CIFAR-10 image and label from raw bytes """
    # convert bytes to vector of uint8 that is record_bytes long
    record_vector = tf.decode_raw(raw_record, tf.uint8)

    # first byte is label, will convert to uint32 then one-hot
    label = tf.cast(record_vector[0], tf.int32)
    label = tf.one_hot(label, _NUM_CLASSES)

    # remaining bytes we reshape from
    # [ depth * height * width ] to [ depth, height, width ]
    depth_major = tf.reshape(record_vector[1:_RECORD_BYTES],
                             [_NUM_CHANNELS, _HEIGHT, _WIDTH])

    # reshape from [depth, height, width] to [height, width, depth]
    image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

    image = preprocess_image(image, is_training)

    return image, label

def preprocess_image(image, is_training):
    """ Preprocess a single image of layout [height, width, depth] """
    if is_training:
        # resize the image to add extra pixels on each side
        image = tf.image.resize_image_with_crop_or_pad(
            image, _HEIGHT + 8, _WIDTH + 8)

        # crop a [_HEIGHT, _WIDTH] section randomly
        image = tf.random_crop(image, [_HEIGHT, _WIDTH, _NUM_CHANNELS])

        # flip image horizontally randomly
        image = tf.image.random_flip_left_right(image)

    # Subtract off mean and divide by variance of the pixels
    image = tf.image.per_image_standardization(image)
    return image

def input_fn(is_training, data_dir, batch_size, num_epochs=1,
             num_parallel_calls=1):
    """ Input_fn using the tf.data input pipeline for the CIFAR-10 Dataset

    Args:
        is_training: boolean denoting whether input is for training
        data_dir: directory where input data is
        batch_size: number of samples per batch
        num_epochs: total number of epochs to repeat dataset
        num_parallel_calls: number of records processed in parallel
            this can be optimized per dataset but shshould be
            aprrox number of available CPU cores

    Returns:
        dataset that can be used for iteration
    """
    filenames = get_filnames(is_training, data_dir)
    dataset = tf.data.FixedLengthRecordDataset(filenames, _RECORD_BYTES)

    return resnet.process_record_dataset(dataset, is_training, batch_size,
                                         _NUM_IMAGES['train'], parse_record,
                                         num_epochs, num_parallel_calls)

##############################################################################
# Running the model itself
##############################################################################
class Cifar10Model(resnet.Model):

    def __init__(self, resnet_size, data_format=None, num_classes=_NUM_CLASSES):
        """ Parameters that work for CIFAR-10 data ResNet

        Args:
            resnet_size: number of convolutional layers needed for the model
            data_format: either 'channels_first' or 'channels_last'
            num_classes: number of output classes needed for the model
        """
        if resnet_size % 6 != 2:
            raise ValueError('resnet_size must be 6n + 2:', resnet_size)

        num_blocks = (resnet_size - 2) // 6

        super(Cifar10Model, self).__init__(
            resnet_size=resnet_size,
            num_classes=num_classes,
            num_filters=16,
            kernel_size=3,
            conv_stride=1,
            first_pool_size=None,
            first_pool_stride=None,
            second_pool_size=8,
            second_pool_stride=1,
            block_fn=resnet.building_block,
            block_sizes=[num_blocks] * 3,
            block_strides=[1, 2, 2],
            final_size=64,
            data_format=data_format)

def cifar10_model_fn(features, labels, mode, params):
    """ Model function for CIFAR10 resnet """
    features = tf.reshape(features, [-1, _HEIGHT, _WIDTH, _NUM_CHANNELS])

    learning_rate_fn = resnet.learning_rate_with_decay(
        batch_size=params['batch_size'], batch_denom=128,
        num_images=_NUM_IMAGES['train'], boundary_epochs=[100, 150, 200],
        decay_rates=[1, 0.1, 0.01, 0.001])

    # Going to use a weight decay of .0002 since that is what tesnorflow
    # official did
    weight_decay = 2e-4

    # According to official tensorflow, using all variables for weight decay
    # was the best
    # so define a loss filter that includes all variables
    def loss_filer_fn(name):
        return True

    return resnet.resnet_model_fn(features, labels, mode, Cifar10Model,
                                resnet_size=params['resnet_size'],
                                weight_decay=weight_decay,
                                learning_rate_fn=learning_rate_fn,
                                momentum=0.9,
                                data_format=params['data_format'],
                                loss_filter_fn=loss_filer_fn)

def main(unused_argv):
    resnet.resnet_main(FLAGS, cifar10_model_fn, input_fn)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    parser = resnet.ResnetArgParser()

    # set defaults
    parser.set_defaults(data_dir='/tmp/cifar10_data',
                        model_dir='/tmp/cifar10_model',
                        resnet_size=32,
                        train_epochs=250,
                        epochs_per_eval=10,
                        batch_size=128)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(argv=[sys.argv[0]] + unparsed)
