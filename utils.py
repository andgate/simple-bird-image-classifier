import tensorflow as tf


@tf.function
def normalize_img(image, mean, std):
    image = (tf.cast(image, tf.float32) - mean) / std
    return image


@tf.function
def denormalize_img(image, mean, std):
    image = image * std + mean
    return image


@tf.function
def crop_to_bounding_box(image, bbox):
    y_min = bbox[0]
    x_min = bbox[1]
    y_max = bbox[2]
    x_max = bbox[3]
    image_shape = tf.cast(tf.shape(image), tf.float32)
    image_height = image_shape[0]
    image_width = image_shape[1]
    offset_height = y_min * image_height
    offset_width = x_min * image_width
    target_height = (y_max - y_min) * image_height
    target_width = (x_max - x_min) * image_width
    image = tf.image.crop_to_bounding_box(
        image,
        tf.cast(offset_height, tf.int32),
        tf.cast(offset_width, tf.int32),
        tf.cast(target_height, tf.int32),
        tf.cast(target_width, tf.int32),
    )
    return image


@tf.function
# Data preprocessing functions
def resize_with_pad_img(image, target_height, target_width):
    return tf.image.resize_with_pad(
        image,
        target_height,
        target_width,
    )
