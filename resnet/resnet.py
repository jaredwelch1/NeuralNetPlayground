"""Contains model definitions for a residual network using batch normalization
before every weight layer. This is the resnet v2 architecture that can be found
here: https://arxiv.org/abs/1603.05027"""

import os
import tensorflow as tf



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
    if projection_shortcut is not None:
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

def block_layer(inputs, filters, block_fn, blocks, strides, training, name,
                data_format):
    """Creates a layer of blocks for resnet model

    Args:
        inputs: tensor of size [batch, channels, height_in, width_in] or
            [batch, height_in, width_in, channels]
        filters: number of filters for 1st convolutional layer
        block_fn: helper function for block type
            either bottleneck or building_block
        blocks: number of blocks in the layer
        strides: stride to use for the first convolution of the layer
            if greater than 1, will downsample input
        training: True or False whether currently training
        name: string name for the tensor output of the block layer
        data_format: either channels_first or channels_last

    Returns:
        output tensor of the block layer
    """

    # this line would need to be modified if bottleneck block implemented
    filters_out = filters

    def projection_shortcut(inputs):
        """ helper function for projection between layers """
        return conv2d_fixed_padding(
            inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
            data_format=data_format)

    # first block in layer will use projection shortcut and strides
    inputs = block_fn(inputs, filters, training, projection_shortcut, strides,
                      data_format)

    for _ in range(1, blocks):
        inputs = block_fn(inputs, filters, training, None, 1, data_format)

    return tf.identity(inputs, name)


class Model(object):
    """Base class for resnet v2 model"""

    def __init__(self, resnet_size, num_classes, num_filters, kernel_size,
                 conv_stride, first_pool_size, first_pool_stride,
                 second_pool_size, second_pool_stride, block_fn, block_sizes,
                 block_strides, final_size, data_format=None):
        """Creates a model for classifying images

        Args:
            resnet_size: single integer for size of Model
            num_classes: number of classes used as labels
            num_filters: number of filters to use for the first
                block layer, doubled for each subsequent layer
            kernel_size: size of convolution kernel
            conv_stride: stride size for initial convolutional layer
            first_pool_size: pool size for first pooling layer
                if None, first pooling is skipped
            first_pool_stride: stride size for first pooling layer
                not used if first_pool_size is None
            second_pool_size: pool size for second pooling layer
            second_pool_stride: stride size for final pooling layer
            block_fn: function to use for block layer, either bottleneck or
                bulding block
            block_sizes: list containing n values, where n is the count for
                block layers, with the ith value equal to number of blocks in
                that layer
            block_strides: list of integers representing the stride size for
                each layer of blocks. should be same length as block_sizes
            final_size: expected size of model after second pooling
            data_format: input format ('channels_last', 'channels_first', None)
                if None, format depends on GPU availability
            """
        self.resnet_size = resnet_size

        if not data_format:
            data_format = (
                'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

        self.data_format = data_format
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.conv_stride = conv_stride
        self.first_pool_size = first_pool_size
        self.first_pool_stride = first_pool_stride
        self.second_pool_size = second_pool_size
        self.second_pool_stride = second_pool_stride
        self.block_fn = block_fn
        self.block_sizes = block_sizes
        self.block_strides = block_strides
        self.final_size = final_size

    def __call__(self, inputs, training):
        """Add operations for classifying a batch of input images

        Args:
            inputs: tensor representing a batch of input images
            training: bool that when True adds operations needed for training

        Returns:
            A logits tensor with shape [<batch_size>, self.num_classes]
        """

        if self.data_format == 'channels_first':
            # convert input from channels_last to channels_first
            # according to tensorflow/models/oficial/resnet
            # this increases performance for GPU
            inputs = tf.transpose(inputs, [0, 3, 1, 2])

        inputs = conv2d_fixed_padding(
            inputs=inputs, filters=self.num_filters, kernel_size=self.kernel_size,
            strides=self.conv_stride, data_format=self.data_format)
        inputs = tf.identity(inputs, 'initial_conv')

        if self.first_pool_size:
            inputs = tf.layers.max_pooling2d(
                inputs=inputs, pool_size=self.first_pool_size,
                strides=self.first_pool_stride, padding='SAME',
                data_format=self.data_format)
            inputs = tf.identity(inputs, 'initial_max_pool')

        # after first layer of pooling, add block layers
        for i, num_blocks in enumerate(self.block_sizes):
            # increases each layer by doubling, so calc for each layer
            num_filters = self.num_filters * (2**i)
            inputs = block_layer(
                inputs=inputs, filters=num_filters, block_fn=self.block_fn,
                blocks=num_blocks, strides=self.block_strides[i],
                training=training, name='block_layer{}'.format(i + 1),
                data_format=self.data_format)

        # batch norm again before final pooling
        inputs = batch_norm_relu(inputs, training, self.data_format)
        inputs = tf.layers.average_pooling2d(
            inputs=inputs, pool_size=self.second_pool_size,
            strides=self.second_pool_stride, padding='VALID',
            data_format=self.data_format)
        inputs = tf.identity(inputs, 'final_avg_pool')

        # final dense read-out layer
        inputs = tf.reshape(inputs, [-1, self.final_size])
        inputs = tf.layers.dense(inputs=inputs, units=self.num_classes)
        inputs = tf.identity(inputs, 'final_dense')
        return inputs
