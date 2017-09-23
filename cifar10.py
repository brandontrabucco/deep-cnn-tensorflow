import tensorflow as tf


# The location on the disk of project
PROJECT_BASEDIR = ("C:/Users/brand/Google Drive/" +
    "Academic/Research/Deep Convolutional Image Classification/")


# The location on the disk of checkpoints
CHECKPOINT_BASEDIR = (PROJECT_BASEDIR + "Checkpoints/")


# The location on the disk of cifar-10 dataset binaries.
DATASET_BASEDIR_CIFAR10 = ("C:/Users/brand/Google Drive/" +
    "Academic/Research/Datasets/cifar-10-binary/")


# The location on the disk of cifar-100 dataset binaries.
DATASET_BASEDIR_CIFAR100 = ("C:/Users/brand/Google Drive/" +
    "Academic/Research/Datasets/cifar-100-binary/")


# To be converted to file queue with TensorFlow
DATASET_FILENAMES_CIFAR10 = [
    DATASET_BASEDIR_CIFAR10 + "data_batch_1.bin",
    DATASET_BASEDIR_CIFAR10 + "data_batch_2.bin",
    DATASET_BASEDIR_CIFAR10 + "data_batch_3.bin",
    DATASET_BASEDIR_CIFAR10 + "data_batch_4.bin",
    DATASET_BASEDIR_CIFAR10 + "data_batch_5.bin",
    DATASET_BASEDIR_CIFAR10 + "test_batch.bin"]


# Locate dataset files on hard disk
for FILE_CIFAR10 in DATASET_FILENAMES_CIFAR10:
    if not tf.gfile.Exists(FILE_CIFAR10):
        raise ValueError('Failed to find file: ' + FILE_CIFAR10)


# To be converted to file queue with TensorFlow
DATASET_FILENAMES_CIFAR100 = [
    DATASET_BASEDIR_CIFAR100 + "train.bin",
    DATASET_BASEDIR_CIFAR100 + "test.bin"]


# Locate dataset files on hard disk
for FILE_CIFAR100 in DATASET_FILENAMES_CIFAR100:
    if not tf.gfile.Exists(FILE_CIFAR100):
        raise ValueError('Failed to find file: ' + FILE_CIFAR100)


# Specify the binary encoding for cifar-10
LABEL_BYTES_CIFAR10 = 1
IMAGE_HEIGHT_CIFAR10 = 32
IMAGE_WIDTH_CIFAR10 = 32
IMAGE_CHANNELS_CIFAR10 = 3


# Specify the binary encoding for cifar-100
LABEL_BYTES_CIFAR100 = 2
IMAGE_HEIGHT_CIFAR100 = 32
IMAGE_WIDTH_CIFAR100 = 32
IMAGE_CHANNELS_CIFAR100 = 3


# Number examples in dataset
DATASET_SIZE = 50000
TESTING_SIZE = 10000


# Number of simultaneous examples
BATCH_SIZE = 500


# Number steps per epoch
EPOCH_SIZE = DATASET_SIZE // BATCH_SIZE
TEST_EPOCH_SIZE = TESTING_SIZE // BATCH_SIZE


# Parallel image processing threads
NUM_THREADS = 8


def decode_record_cifar(filename_queue, label_bytes, image_height=32, image_width=32, image_channels=3):

    # Read single record from cifar-10
    DATASET_READER = tf.FixedLengthRecordReader(
        record_bytes=(image_height * image_width * image_channels + label_bytes))
    key, value_bytes = DATASET_READER.read(filename_queue)


    # Decode binary value string
    value_bytes = tf.decode_raw(value_bytes, tf.uint8)


    # Decode label from single record bytes
    label = tf.reshape(
            tf.strided_slice(value_bytes, [0], [label_bytes]),
        [label_bytes])
    label = tf.cast(label, tf.float32)[-1]


    # Decode image from single record bytes
    image = tf.reshape(
        tf.strided_slice(value_bytes, [label_bytes],
            [(image_height * image_width * image_channels + label_bytes)]),
        [image_channels, image_height, image_width])


    # Convert from [depth, height, width] to [height, width, depth].
    image = tf.transpose(image, [1, 2, 0])
    image = tf.cast(image, tf.float32)

    return image, label


def preprocess_image(image, crop_height=32, crop_width=32, normalize_image=True):

    if crop_height < 32 or crop_width < 32:

        # Randomly crop a [height, width] section of the image.
        image = tf.random_crop(image, [crop_height, crop_width, 3])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)


    if normalize_image:

        # Subtact mean and divide variance of pixels
        image = tf.image.per_image_standardization(image)

    return image


def generate_batch(image, label, batch_size=32, num_threads=4, shuffle_batch=True):

    # Shuffle batch randomly
    if shuffle_batch:

        # Construct batch from queue of records
        image_batch, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=50000,
            min_after_dequeue=10000)


    # Preserve order of batch
    else:

        # Construct batch from queue of records
        image_batch, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=50000)

    return image_batch, tf.reshape(label_batch, [batch_size])


# Generate single training batch of images and labels cifar-10
def training_batch_cifar10(crop_height=32, crop_width=32, normalize_image=True, shuffle_batch=True):

    # Enforce proper crop dimensions
    assert crop_height > 0 and crop_height <= 32, "Invalid crop dimension: height"
    assert crop_width > 0 and crop_width <= 32, "Invalid crop dimension: width"  


    # To be converted to file queue with TensorFlow
    dataset_filenames = tf.constant(DATASET_FILENAMES_CIFAR10[:-1])


    # A queue to generate batches of size 32 across all files
    filename_queue = tf.train.string_input_producer(dataset_filenames)


    # Decode from string to floating point
    image, label = decode_record_cifar(
        filename_queue, 
        LABEL_BYTES_CIFAR10, 
        image_height=IMAGE_HEIGHT_CIFAR10, 
        image_width=IMAGE_WIDTH_CIFAR10, 
        image_channels=IMAGE_CHANNELS_CIFAR10)


    # Crop and sharpen image
    image = preprocess_image(image, crop_height, crop_width, normalize_image)


    # Combine image and label queue into batch
    return generate_batch(
        image, 
        label, 
        batch_size=BATCH_SIZE, 
        num_threads=NUM_THREADS, 
        shuffle_batch=shuffle_batch)


# Generate single testing batch of images and labels cifar-10
def testing_batch_cifar10(crop_height=32, crop_width=32, normalize_image=True, shuffle_batch=True):

    # Enforce proper crop dimensions
    assert crop_height > 0 and crop_height <= 32, "Invalid crop dimension: height"
    assert crop_width > 0 and crop_width <= 32, "Invalid crop dimension: width"  


    # To be converted to file queue with TensorFlow
    dataset_filenames = tf.constant(DATASET_FILENAMES_CIFAR10[-1:])


    # A queue to generate batches of size 32 across all files
    filename_queue = tf.train.string_input_producer(dataset_filenames)


    # Decode from string to floating point
    image, label = decode_record_cifar(
        filename_queue, 
        LABEL_BYTES_CIFAR10, 
        image_height=IMAGE_HEIGHT_CIFAR10, 
        image_width=IMAGE_WIDTH_CIFAR10, 
        image_channels=IMAGE_CHANNELS_CIFAR10)


    # Crop and sharpen image
    image = preprocess_image(image, crop_height, crop_width, normalize_image)


    # Combine image and label queue into batch
    return generate_batch(
        image, 
        label, 
        batch_size=BATCH_SIZE, 
        num_threads=NUM_THREADS, 
        shuffle_batch=shuffle_batch)


# Generate single training batch of images and labels cifar-100
def training_batch_cifar100(crop_height=32, crop_width=32, normalize_image=True, shuffle_batch=True):

    # Enforce proper crop dimensions
    assert crop_height > 0 and crop_height <= 32, "Invalid crop dimension: height"
    assert crop_width > 0 and crop_width <= 32, "Invalid crop dimension: width"


    # To be converted to file queue with TensorFlow
    dataset_filenames = tf.constant(DATASET_FILENAMES_CIFAR100[:-1])


    # A queue to generate batches of size 32 across all files
    filename_queue = tf.train.string_input_producer(dataset_filenames)


    # Decode from string to floating point
    image, label = decode_record_cifar(
        filename_queue, 
        LABEL_BYTES_CIFAR100, 
        image_height=IMAGE_HEIGHT_CIFAR100, 
        image_width=IMAGE_WIDTH_CIFAR100, 
        image_channels=IMAGE_CHANNELS_CIFAR100)


    # Crop and sharpen image
    image = preprocess_image(image, crop_height, crop_width, normalize_image)


    # Combine image and label queue into batch
    return generate_batch(
        image, 
        label, 
        batch_size=BATCH_SIZE, 
        num_threads=NUM_THREADS, 
        shuffle_batch=shuffle_batch)


# Generate single testing batch of images and labels cifar-100
def testing_batch_cifar100(crop_height=32, crop_width=32, normalize_image=True, shuffle_batch=True):

    # Enforce proper crop dimensions
    assert crop_height > 0 and crop_height <= 32, "Invalid crop dimension: height"
    assert crop_width > 0 and crop_width <= 32, "Invalid crop dimension: width"


    # To be converted to file queue with TensorFlow
    dataset_filenames = tf.constant(DATASET_FILENAMES_CIFAR100[-1:])


    # A queue to generate batches of size 32 across all files
    filename_queue = tf.train.string_input_producer(dataset_filenames)


    # Decode from string to floating point
    image, label = decode_record_cifar(
        filename_queue, 
        LABEL_BYTES_CIFAR100, 
        image_height=IMAGE_HEIGHT_CIFAR100, 
        image_width=IMAGE_WIDTH_CIFAR100, 
        image_channels=IMAGE_CHANNELS_CIFAR100)


    # Crop and sharpen image
    image = preprocess_image(image, crop_height, crop_width, normalize_image)


    # Combine image and label queue into batch
    return generate_batch(
        image, 
        label, 
        batch_size=BATCH_SIZE, 
        num_threads=NUM_THREADS, 
        shuffle_batch=shuffle_batch)


# Naming conventions
PREFIX_CONVOLUTION = "conv"
PREFIX_POOLING = "pool"
PREFIX_NORMALIZATION = "norm"
PREFIX_DENSE = "dense"
PREFIX_SOFTMAX = "softmax"
PREFIX_TOTAL = "total"


# Naming conventions
EXTENSION_NUMBER = (lambda number: "_" + str(number))
EXTENSION_LOSS = "_loss"
EXTENSION_WEIGHTS = "_weights"
EXTENSION_BIASES = "_biases"
EXTENSION_OFFSET = "_offset"
EXTENSION_SCALE = "_scale"
EXTENSION_ACTIVATION = "_activation"


# Naming conventions
COLLECTION_LOSSES = "losses"
COLLECTION_PARAMETERS = "parameters"
COLLECTION_ACTIVATIONS = "activations"


def initialize_weights_cpu(name, shape, standard_deviation=0.01, decay_factor=None):

    # Force usage of cpu
    with tf.device("/cpu:0"):

        # Sample weights from normal distribution
        weights = tf.get_variable(
            name,
            shape, 
            initializer=tf.truncated_normal_initializer(
                stddev=standard_deviation,
                dtype=tf.float32),
            dtype=tf.float32)

    # Add weight decay to loss function
    if decay_factor is not None:

        # Calculate decay with l2 loss
        weight_decay = tf.multiply(
            tf.nn.l2_loss(weights), 
            decay_factor, 
            name=(name + EXTENSION_LOSS))
        tf.add_to_collection(COLLECTION_LOSSES, weight_decay)

    return weights


def initialize_biases_cpu(name, shape):

    # Force usage of cpu
    with tf.device("/cpu:0"):

        # Sample weights from normal distribution
        biases = tf.get_variable(
            name,
            shape, 
            initializer=tf.constant_initializer(1.0),
            dtype=tf.float32)

    return biases


def inference_cifar10(image_batch):

    # Bind to name for consistency
    activation = image_batch


    # Create scope for first convolution
    with tf.variable_scope(PREFIX_CONVOLUTION + EXTENSION_NUMBER(1)) as scope:

        # Create kernel of weights
        weights = initialize_weights_cpu((scope.name + EXTENSION_WEIGHTS), [3, 3, 3, 32])
        
        # Convolve weight kernel across activation
        activation = tf.nn.conv2d(activation, weights, [1, 1, 1, 1], padding="SAME")
        
        # Create  and add biases
        biases = initialize_biases_cpu((scope.name + EXTENSION_BIASES), [32])
        activation = tf.nn.bias_add(activation, biases)
        
        # Pass through ReLU function
        activation = tf.nn.relu(activation, name=(scope.name + EXTENSION_ACTIVATION))


    # Create scope for first pooling
    with tf.variable_scope(PREFIX_POOLING + EXTENSION_NUMBER(1)) as scope:

        # Reduce [height, width] by factor 2
        activation = tf.nn.max_pool(
            activation, 
            ksize=[1, 3, 3, 1], 
            strides=[1, 2, 2, 1], 
            padding="SAME", 
            name=(scope.name + EXTENSION_ACTIVATION))


    # Create scope for first normalization
    with tf.variable_scope(PREFIX_NORMALIZATION + EXTENSION_NUMBER(1)) as scope:

        # Calculate mean and variance
        mean, variance = tf.nn.moments(activation, [0, 1, 2])


        # Create parameters to reverse normalization
        offset = initialize_biases_cpu((scope.name + EXTENSION_OFFSET), [32])
        scale = initialize_biases_cpu((scope.name + EXTENSION_SCALE), [32])


        # Perform global normalization
        activation = tf.nn.batch_normalization(activation, mean, variance, offset, scale, 1e-3)


    # Create scope for second convolution
    with tf.variable_scope(PREFIX_CONVOLUTION + EXTENSION_NUMBER(2)) as scope:

        # Create kernel of weights
        weights = initialize_weights_cpu((scope.name + EXTENSION_WEIGHTS), [3, 3, 32, 64])
        
        # Convolve weight kernel across activation
        activation = tf.nn.conv2d(activation, weights, [1, 1, 1, 1], padding="SAME")
        
        # Create  and add biases
        biases = initialize_biases_cpu((scope.name + EXTENSION_BIASES), [64])
        activation = tf.nn.bias_add(activation, biases)
        
        # Pass through ReLU function
        activation = tf.nn.relu(activation, name=(scope.name + EXTENSION_ACTIVATION))


    # Create scope for second pooling
    with tf.variable_scope(PREFIX_POOLING + EXTENSION_NUMBER(2)) as scope:

        # Reduce [height, width] by factor 2
        activation = tf.nn.max_pool(
            activation, 
            ksize=[1, 3, 3, 1], 
            strides=[1, 2, 2, 1], 
            padding="SAME", 
            name=(scope.name + EXTENSION_ACTIVATION))


    # Create scope for second normalization
    with tf.variable_scope(PREFIX_NORMALIZATION + EXTENSION_NUMBER(2)) as scope:

        # Calculate mean and variance
        mean, variance = tf.nn.moments(activation, [0, 1, 2])


        # Create parameters to reverse normalization
        offset = initialize_biases_cpu((scope.name + EXTENSION_OFFSET), [64])
        scale = initialize_biases_cpu((scope.name + EXTENSION_SCALE), [64])


        # Perform global normalization
        activation = tf.nn.batch_normalization(activation, mean, variance, offset, scale, 1e-3)


    # Create scope for third convolution
    with tf.variable_scope(PREFIX_CONVOLUTION + EXTENSION_NUMBER(3)) as scope:

        # Create kernel of weights
        weights = initialize_weights_cpu((scope.name + EXTENSION_WEIGHTS), [3, 3, 64, 128])
        
        # Convolve weight kernel across activation
        activation = tf.nn.conv2d(activation, weights, [1, 1, 1, 1], padding="SAME")
        
        # Create  and add biases
        biases = initialize_biases_cpu((scope.name + EXTENSION_BIASES), [128])
        activation = tf.nn.bias_add(activation, biases)
        
        # Pass through ReLU function
        activation = tf.nn.relu(activation, name=(scope.name + EXTENSION_ACTIVATION))


    # Create scope for third pooling
    with tf.variable_scope(PREFIX_POOLING + EXTENSION_NUMBER(3)) as scope:

        # Reduce [height, width] by factor 2
        activation = tf.nn.max_pool(
            activation, 
            ksize=[1, 3, 3, 1], 
            strides=[1, 2, 2, 1], 
            padding="SAME", 
            name=(scope.name + EXTENSION_ACTIVATION))


    # Create scope for third normalization
    with tf.variable_scope(PREFIX_NORMALIZATION + EXTENSION_NUMBER(3)) as scope:

        # Calculate mean and variance
        mean, variance = tf.nn.moments(activation, [0, 1, 2])


        # Create parameters to reverse normalization
        offset = initialize_biases_cpu((scope.name + EXTENSION_OFFSET), [128])
        scale = initialize_biases_cpu((scope.name + EXTENSION_SCALE), [128])


        # Perform global normalization
        activation = tf.nn.batch_normalization(activation, mean, variance, offset, scale, 1e-3)


    # Reshape to matrix for dense layers
    activation = tf.reshape(activation, [BATCH_SIZE, -1])


    # Create scope for fourth dense layer
    with tf.variable_scope(PREFIX_DENSE + EXTENSION_NUMBER(4)) as scope:

        # Create matrix of weights
        weights = initialize_weights_cpu((scope.name + EXTENSION_WEIGHTS), [2048, 1024])


        # Multiple weight matrix by activation
        activation = tf.matmul(activation, weights)
        

        # Create  and add biases
        biases = initialize_biases_cpu((scope.name + EXTENSION_BIASES), [1024])
        activation = tf.nn.bias_add(activation, biases)

        
        # Pass through ReLU function
        activation = tf.nn.relu(activation, name=(scope.name + EXTENSION_ACTIVATION))


    # Create scope for fifth dense layer
    with tf.variable_scope(PREFIX_DENSE + EXTENSION_NUMBER(5)) as scope:

        # Create matrix of weights
        weights = initialize_weights_cpu((scope.name + EXTENSION_WEIGHTS), [1024, 512])


        # Multiple weight matrix by activation
        activation = tf.matmul(activation, weights)
        

        # Create  and add biases
        biases = initialize_biases_cpu((scope.name + EXTENSION_BIASES), [512])
        activation = tf.nn.bias_add(activation, biases)

        
        # Pass through ReLU function
        activation = tf.nn.relu(activation, name=(scope.name + EXTENSION_ACTIVATION))


    # Create scope for sixth dense layer
    with tf.variable_scope(PREFIX_DENSE + EXTENSION_NUMBER(6)) as scope:

        # Create matrix of weights
        weights = initialize_weights_cpu((scope.name + EXTENSION_WEIGHTS), [512, 10])


        # Multiple weight matrix by activation
        activation = tf.matmul(activation, weights)
        

        # Create  and add biases
        biases = initialize_biases_cpu((scope.name + EXTENSION_BIASES), [10])
        activation = tf.nn.bias_add(activation, biases)

    # Note the final layer has no activation function
    return activation


def loss(prediction, labels):

    # Labels will be used as indeces for cross entropy
    labels = tf.cast(labels, tf.int32)
    

    # Calculate softmax cross entropy loss
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels,
        logits=prediction,
        name=PREFIX_SOFTMAX)


    # Calculate cross entropy mean
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name=(PREFIX_SOFTMAX + EXTENSION_LOSS))
    tf.add_to_collection(COLLECTION_LOSSES, cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'), name=(PREFIX_TOTAL + EXTENSION_LOSS))


# Hyperparameters 
INITIAL_LEARNING_RATE = 0.001
DECAY_STEPS = 10 * EPOCH_SIZE
DECAY_FACTOR = 0.95


def train(total_loss):

    # Keep track of current training step
    global_step = tf.train.get_or_create_global_step()


    # Decay learning rate
    learning_rate = tf.train.exponential_decay(
        INITIAL_LEARNING_RATE, 
        global_step, 
        DECAY_STEPS, 
        DECAY_FACTOR, 
        staircase=True)


    # Create optimizer for gradient updates
    optimizer = tf.train.AdamOptimizer(learning_rate)


    # Minimize loss using optimizer
    gradient = optimizer.minimize(total_loss, global_step=global_step)

    return gradient


# Logging controls
LOG_PERIOD = EPOCH_SIZE // 4


def train_cifar10(num_epoch=1):

    # Watch compute time per batch
    from time import time
    from datetime import datetime


    # Convert epoch to batch steps
    num_steps = num_epoch * EPOCH_SIZE


    # Create new graph
    with tf.Graph().as_default():

        # Connect cifar-10 dataset
        image_batch, label_batch = training_batch_cifar10()


        # Compute prediction
        prediction_batch = inference_cifar10(image_batch)
        index_batch = tf.reshape(tf.argmax(prediction_batch, axis=1), [BATCH_SIZE])
        
        
        # Compute cross entropy loss
        loss_batch = loss(prediction_batch, label_batch)


        # Update trainable parameters
        gradient_batch = train(loss_batch)


        # Track loss to generate plot
        data_points = []


        # Report training progress
        class LogProgressHook(tf.train.SessionRunHook):

            # Session is initialized
            def begin(self):
                self.current_step = 0
                self.batch_speed = 0

            # Just before inference
            def before_run(self, run_context):
                self.current_step += 1
                self.start_time = time()
                return tf.train.SessionRunArgs([
                    label_batch, index_batch, loss_batch])

            # Just after inference
            def after_run(self, run_context, run_values):
                
                # Calculate weighted speed
                self.batch_speed = (0.2 * self.batch_speed) + (0.8 * (1.0 / (time() - self.start_time + 1e-7)))


                # Update every period of steps
                if (self.current_step % EPOCH_SIZE == 0):

                    # Obtain graph results
                    label_value, index_value, loss_value = run_values.results


                    # Calculate accuracy across batch
                    correct_labels = 0
                    for i in range(BATCH_SIZE):
                        if label_value[i] == index_value[i]:
                            correct_labels += 1


                    # Display date, batch speed, estimated time, loss, and accuracy
                    print(
                        datetime.now(),
                        "CUR: %d" % self.current_step,
                        "REM: %d" % (num_steps - self.current_step),
                        "SPD: %.2f bat/sec" % self.batch_speed,
                        "ETA: %.2f hrs" % ((num_steps - self.current_step) / self.batch_speed / 60 / 60),
                        "L: %.2f" % loss_value,
                        "A: %.2f %%" % (correct_labels / BATCH_SIZE * 100))


                    # Record current loss
                    data_points.append(loss_value)


        # Prepare to save and load models
        model_saver = tf.train.Saver()


        # Create new session for graph
        with tf.train.MonitoredTrainingSession(hooks=[
            tf.train.StopAtStepHook(num_steps=num_steps), 
            tf.train.CheckpointSaverHook(CHECKPOINT_BASEDIR, save_steps=EPOCH_SIZE, saver=model_saver),
            LogProgressHook()]) as session:

            # Repeat training iteratively
            while not session.should_stop():

                # Run single batch of training
                session.run(gradient_batch)


    # Construct and save plot
    import matplotlib.pyplot as plt
    plt.plot(data_points)
    plt.xlabel("Training Epoch")
    plt.ylabel("Mean Cross-Entropy Loss")
    plt.savefig(datetime.now().strftime("%Y_%B_%d_%H_%M_%S") + "_training_loss.png")
    plt.close()


def test_cifar10(model_checkpoint):

    # Watch compute time per batch
    from time import time
    from datetime import datetime


    # Number of epoch in
    num_steps = TEST_EPOCH_SIZE * 10


    # Calculate the final accuracy
    final_accuracy = 0.0


    # Create new graph
    with tf.Graph().as_default():

        # Connect cifar-10 dataset
        image_batch, label_batch = testing_batch_cifar10()


        # Compute prediction
        prediction_batch = inference_cifar10(image_batch)
        index_batch = tf.reshape(tf.argmax(prediction_batch, axis=1), [BATCH_SIZE])

        
        # Compute cross entropy loss
        loss_batch = loss(prediction_batch, label_batch)


        # Report testing progress
        class LogProgressHook(tf.train.SessionRunHook):

            # Just before inference
            def before_run(self, run_context):
                return tf.train.SessionRunArgs([
                    label_batch, index_batch])

            # Just after inference
            def after_run(self, run_context, run_values):

                # Obtain graph results
                label_value, index_value = run_values.results


                # Calculate accuracy across batch
                correct_labels = 0
                for i in range(BATCH_SIZE):
                    if label_value[i] == index_value[i]:
                        correct_labels += 1


                # Sum acuracies across each batch
                nonlocal final_accuracy
                final_accuracy += (correct_labels / BATCH_SIZE * 100)


        # Prepare to save and load models
        model_saver = tf.train.Saver()


        # Create new session for graph
        with tf.train.MonitoredTrainingSession(hooks=[
            LogProgressHook()]) as session:

            # Load progress from checkpoint
            model_saver.restore(session, model_checkpoint)


            # Repeat training iteratively
            i = 1
            while i < num_steps:
                i += 1

                # Run single batch of testing
                session.run(loss_batch)


    print("Final Accuracy: %f" % (final_accuracy / num_steps))