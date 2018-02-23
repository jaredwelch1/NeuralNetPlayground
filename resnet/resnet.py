"""Contains model definitions for a residual network using batch normalization
before every weight layer. This is the resnet v2 architecture that can be found
here: https://arxiv.org/abs/1603.05027"""

import argparse
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


###############################################################################
# Functions for trainng/evaluation/validation loops
###############################################################################
def learning_rate_with_decay(
        batch_size, batch_denom, num_images, boundary_epochs, decay_rates):
    """Get a learning rate that decays as training progresses

    Args:
        batch_size: number of images in each training batch
        batch_denom: this value will be used to scale learning rate
            '0.1 * batch_size' is divided by this number, such that
            the initial learning rate is 0.1
        num_images: total count of images in training data
        boundary_epochs: list of ints representing the epochs where learning
            rate decays
        decay_rates: list of floats representing the decay rates to be used
            for scaling learning rate. same length as boundary_epochs

    Returns:
        returns function that takes single argument, the number of batches
        trained so far (global_step) and returns learning rate to be used
        for the next batch
    """
    initial_learning_rate = 0.1 * batch_size / batch_denom
    batches_per_epoch = num_images / batch_size

    # multiply learning rate by 0.1 at each boundary epoch
    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(global_step):
        global_step = tf.cast(global_step, tf.int32)
        return tf.train.piecewise_constant(global_step, boundaries, vals)

    return learning_rate_fn

def resnet_model_fn(features, labels, mode, model_class,
                    resnet_size, weight_decay, learning_rate_fn, momentum,
                    data_format, loss_filter_fn=None):
    """Shared functionality for different resnet model functions

    Inits the ResnetModel representing the model layers and uses it
    to build the EstimatorSpecs for the 'mode'. For training, this
    includes losses, optimizer, and training operations that get passed in

    Args:
        features: tensor representing input images
        labels: tensor representing the class labels
        mode: current estimator mode, should be one of
            'tf.estimator.ModeKeys.TRAIN', 'EVALUATE', 'PREDICT'
        model_class: class representing a tensorflow model that
            has a __call__ method, assumed to be a subclass of ResNet Model
        resnet_size: int for size of model (count of NN layers)
        weight_decay: weight decay loss rate used to regularize learned variables
        learning_rate_fn: function that returns current learning rate
            based on global_step
        momentum: momentum term used for optimization
        data_format: input format for the data
            ('channels_first' or 'channels_last')
        loss_filter_fn: function that takes a string variable name
            and returns whether it shoud be used for loss calculation

    Returns:
        EstimatorSpec parameterized according to the input params and mode
    """

    # Generate a summary node for images
    tf.summary.image('images', features, max_outputs=6)

    model = model_class(resnet_size, data_format)
    logits = model(features, mode == tf.estimator.ModeKeys.TRAIN)

    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate our loss
    cross_entropy = tf.losses.softmax_cross_entropy(
        logits=logits, onehot_labels=labels)

    # create tensor for logging cross_entropy
    tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

    # define a default behavior if a loss_filter_fn is not provided
    # which discludes batch_norm variables
    if not loss_filter_fn:
        def loss_filter_fn(name): # pylint: disable=E0102
            return 'batch_normalization' not in name

    loss = cross_entropy + weight_decay * tf.add_n(
        [tf.nn.l2_loss(v) for v in tf.trainable_variables()
         if loss_filter_fn(v.name)])

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()

        learning_rate = learning_rate_fn(global_step)

        # create tensor for learning rate for logging
        tf.identity(learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)

        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=momentum)

        # batch norm requires update operations to be added as a dependency
        # to train_op
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step)
    else:
        train_op = None

    accuracy = tf.metrics.accuracy(
        tf.argmax(labels, axis=1), predictions['classes'])
    metrics = {'accuracy': accuracy}

    # create tensor for logging training accuracy
    tf.identity(accuracy[1], name='train_accuracy')
    tf.summary.scalar('train_accuracy', accuracy[1])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)

def resnet_main(flags, model_function, input_function):
    """A main for executing resnet model functionality"""
    # using winograd non-fused algorithms gives a small boost in performance
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    # set up a runconfig to save chekcpoints once per train cycle
    run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1e9)
    classifier = tf.estimator.Estimator(
        model_fn=model_function, model_dir=flags.model_dir, config=run_config,
        params={
            'resnet_size': flags.resnet_size,
            'data_format': flags.data_format,
            'batch_size': flags.batch_size
        })

    for _ in range(flags.train_epochs // flags.epochs_per_eval):
        tensors_to_log = {
            'learning_rate': 'learning_rate',
            'cross_entropy': 'cross_entropy',
            'train_accuracy': 'train_accuracy'
        }

        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=100)

        print('Starting a training cycle')

        def input_fn_train():
            return input_function(True, flags.data_dir, flags.batch_size,
                                  flags.epochs_per_eval, flags.num_parallel_calls)

        classifier.train(input_fn=input_fn_train, hooks=[logging_hook])

        print('Starting to evaluate')
        # evaluate the model and print results
        def input_fn_eval():
            return input_function(False, flags.data_dir, flags.batch_size,
                                  1, flags.num_parallel_calls)
        eval_results = classifier.evaluate(input_fn=input_fn_eval)
        print(eval_results)

class ResnetArgParser(argparse.ArgumentParser):
    """Args for configuring and running ResNet model"""
    def __init__(self, resnet_size_choices=None):
        super(ResnetArgParser, self).__init__()
        self.add_argument(
            '--data_dir', type=str, default='/tmp/resnet_data',
            help='The directory where the input data is stored.')

        self.add_argument(
            '--num_parallel_calls', type=int, default=5,
            help='The number of records that are processed in parallel'
            'during input processing.')

        self.add_argument(
            '--model_dir', type=str, default='/tmp/resnet_model',
            help='Directory for storing the model.')

        self.add_argument(
            '--resnet_size', type=int, default=50,
            help='Size of resnet model to use.')

        self.add_argument(
            '--train_epochs', type=int, default=100,
            help='Number of training epochs to run between evaluations.')

        self.add_argument(
            '--epochs_per_eval', type=int, default=1,
            help='Number of training epochs to run between evaluations.')

        self.add_argument(
            '--batch_size', type=int, default=32,
            help='Batch size for training and evaluation.'
        )

        self.add_argument(
            '--data_format', type=str, default=None,
            choices=['channels_first', 'channels_last'],
            help='A flag to override the data format used in the model. '
            'channels_first provides a performance boost on GPU but '
            'is not always compatible with CPU. If left unspecified, '
            'the data format will be chosen automatically based on '
            'whether TensorFlow was built for CPU or GPU.')
