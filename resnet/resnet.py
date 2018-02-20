"""Contains model definitions for a residual network using batch normalization
before every weight layer. This is the resnet v2 architecture that can be found
here: https://arxiv.org/abs/1603.05027"""

import tensorflow as tf
import os


# Constants for batch normalization
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

###############################################################################
# Input processing
###############################################################################
def process_record_dataset(dataset, is_training, batch_size, shuffle_buffer,
                           parse_record_fn, num_epochs=1, num_parallel_calls=1):
    """Given dataset with raw records, parse each record to images and labels,
    return an iterator over the parsed records
    Args:
        dataset: Dataset representing raw records
        is_training: bool for if the input is training data or not
        batch_size: number of samples per batch
        shuffle_buffer: buffer size to use when shuffling records. Larger
            results in better randomness, but smaller has better processing time
        parse_record_fn: a function that takes a raw record in bytes and
            returns (image, label)
        num_epochs: number of epochs for training
        num_parallel_calls: number of records processed in parallel
    Returns:
        Dataset of (image, label) pairs for iteration
    """
    # prefetch a batch at a time
    # load files as we go through shuffling
    dataset = dataset.prefetch(buffer_size=batch_size)
    if is_training:
        # shuffle records before repeating
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)
    # if we train over more than one epoch, repeat Dataset
    dataset = dataset.repeat(num_epochs)

    # parse records into images, labels
    # parse_record_fn will handle mapping bytes the proper form
    dataset = dataset.map(lambda value: parse_record_fn(value, is_training),
                          num_parallel_calls=num_parallel_calls)

    dataset = dataset.batch(batch_size)

    # prefetch here to background the above work
    dataset = dataset.prefetch(1)

    return dataset


###############################################################################
# Model Creation Functions
###############################################################################
def batch_norm_relu(inputs, training, data_format):
    """Performs batch norm then ReLU"""
    # the official tensorflow resnet uses fused=True for performance
    # so I am going to do the same and trust them
    inputs = tf.layers.batch_normalization(
        inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
        scale=True, training=training, fused=True)

    inputs = tf.nn.relu(inputs)
    return inputs

def fixed_padding(inputs, kernel_size, data_format):
    """Pads inputs along spatial dims independently of input size

    Args:
        inputs: tensor, size [batch_size, channels, height_in, width_in]
            or [batch_size, height_in, width_in, channels]
            depending on data_format
        kernel_size: kernel to be used in conv2d or max_pool2d operations
            should be a positive integer
        data_format: input format for the data
            ('channels_first' or 'channels_last')

    Returns:
        tensor with same format as input data either intact (kernel_size is 1)
        or padded (kernel_size greater than 1)
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == 'channels_first':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                        [pad_beg, pad_end], [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]])
    return padded_inputs

def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
    """Strided 2-D convolution with explicit padding."""
    # The padding is consistent and is based only on `kernel_size`, not on the
    # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)

    return tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
        padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer(),
        data_format=data_format)

def building_block(inputs, filters, training, projection_shortcut, strides,
                   data_format):
    """Building block for a residual network with Batch norm before convolutions
    important part of the resnet block is the shortcut function

    Args:
        inputs: tensor of size [batch, channels, height_in, width_in] or
            [batch, height_in, width_in, channels]
        filters: number of filters with convolutions
        training: boolean for whether model is in training or inference mode
        projection_shortcut: function used for projection shortcuts
            (normally 1x1 convolution when downsampling input)
        strides: blocks stride. if greater than 1, will downsample input
        data_format: input format ('channels_first' or 'channels_last')

    Returns:
        output tensor of the block
    """
    shortcut = inputs
    inputs = batch_norm_relu(inputs, training, data_format)

    # Projection shortcut come after first batch norm and ReLU
    # because it performs a 1x1 convolution
    if projection_shortcut:
        shortcut = projection_shortcut(inputs)

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=strides,
        data_format=data_format)

    inputs = batch_norm_relu(inputs, training, data_format)
    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=1,
        data_format=data_format)

    # this line represents the logic that defines residual learning
    # we return not the output of the convolutions from previous layers
    # but instead return shortcut function added to our layer output 
    return inputs + shortcut
