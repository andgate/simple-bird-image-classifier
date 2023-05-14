import tensorflow as tf

# import utils


def calculate_mean_and_std(ds):
    # Initialize accumulators
    mean_accumulator = tf.zeros(3)
    std_accumulator = tf.zeros(3)
    count_accumulator = tf.zeros([])

    ds_metrics = (
        ds.map(lambda x: tf.cast(x["image"], tf.float32))
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )

    for image in ds_metrics:
        # Calculate mean of current image
        current_mean = tf.reduce_mean(image, axis=[0, 1])

        # Update mean accumulator
        current_mean = tf.cast(current_mean, tf.float32)
        mean_accumulator += current_mean

        # Update count
        count_accumulator += 1

    # Calculate mean over entire dataset
    mean = mean_accumulator / count_accumulator

    for image in ds_metrics:
        # Calculate squared difference from the mean
        diff = tf.square(image - mean)

        # Calculate variance of current image
        current_variance = tf.reduce_mean(diff, axis=[0, 1])

        # Update std accumulator
        std_accumulator += current_variance

    # Calculate std over entire dataset
    std = tf.sqrt(std_accumulator / count_accumulator)

    print(f"Mean: {mean.numpy()}, Std: {std.numpy()}")


def calculate_max_image_size(ds):
    img_sizes = []
    for example in ds:
        image = example["image"]
        img_size = tf.constant(image.shape[0:2].as_list())
        img_sizes.append(img_size)

    img_sizes = tf.stack(img_sizes, axis=1)
    max_size = tf.math.reduce_max(img_sizes, axis=1).numpy()
    print(f"max image size {max_size}")


def calculate_max_cropped_size(ds):
    img_sizes = []
    bboxes = []
    for example in ds:
        bbox = example["bbox"]
        image = example["image"]
        img_size = tf.constant(image.shape[0:2].as_list(), dtype=tf.float32)
        img_sizes.append(img_size)
        bboxes.append(bbox)

    img_sizes = tf.stack(img_sizes, axis=1)
    bboxes = tf.stack(bboxes, axis=1)

    h, w = img_sizes
    y_min, x_min, y_max, x_max = bboxes
    # offset_h, offset_w = h * y_min, w * x_min
    target_h = (y_max - y_min) * h
    target_w = (x_max - x_min) * w

    S = tf.stack([target_h, target_w])
    max_target = tf.math.reduce_max(S, axis=1)

    print(f"Maximum crop size {max_target}")
