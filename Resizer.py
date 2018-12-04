import tensorflow as tf
import cv2
import numpy as np


class ResizerSimple:
    def __init__(self, input_width, input_height):
        self.input_width = input_width
        self.input_height = input_height

    def get_resize_func(self, resize_method):
        if resize_method == 'resize_warp':
            return self.resize_warp
        elif resize_method == 'resize_pad_zeros':
            return self.resize_pad_zeros
        elif resize_method == 'resize_lose_part':
            return self.resize_lose_part
        elif resize_method == 'centered_crop':
            return self.centered_crop
        else:
            raise Exception('Resize method not recognized.')

    def resize_warp(self, image):
        image = tf.image.resize_images(image, [self.input_height, self.input_width])
        return image

    def resize_pad_zeros(self, image):
        height, width = tf.shape(image)[0], tf.shape(image)[1]
        scale_height = self.input_height / tf.to_float(height)
        scale_width = self.input_width / tf.to_float(width)
        scale = tf.minimum(scale_height, scale_width)
        size = tf.cast(tf.stack([scale*tf.to_float(height), scale*tf.to_float(width)]), tf.int32)
        image = tf.image.resize_images(image, size)
        image = tf.image.resize_image_with_crop_or_pad(image, self.input_height, self.input_width)
        return image

    def resize_lose_part(self, image):
        height, width = tf.shape(image)[0], tf.shape(image)[1]
        scale_height = self.input_height / tf.to_float(height)
        scale_width = self.input_width / tf.to_float(width)
        scale = tf.maximum(scale_height, scale_width)
        size = tf.cast(tf.stack([scale*tf.to_float(height), scale*tf.to_float(width)]), tf.int32)
        image = tf.image.resize_images(image, size)
        image = tf.image.resize_image_with_crop_or_pad(image, self.input_height, self.input_width)
        return image

    def centered_crop(self, image):
        image = tf.image.resize_image_with_crop_or_pad(image, self.input_height, self.input_width)
        return image


class ResizerWithLabels:
    def __init__(self, input_width, input_height):
        self.input_width = input_width
        self.input_height = input_height

    def get_resize_func(self, resize_method):
        if resize_method == 'resize_warp':
            return self.resize_warp
        elif resize_method == 'resize_pad_zeros':
            return self.resize_pad_zeros
        elif resize_method == 'resize_lose_part':
            raise Exception('Not implemented yet.')
        elif resize_method == 'centered_crop':
            raise Exception('Not implemented yet.')
        else:
            raise Exception('Resize method not recognized.')

    def resize_warp(self, image, label):
        # print('resize_warp')
        # The bounding boxes are in relative coordinates, so they don't need to be changed.
        # image = tf.image.resize_images(image, [self.input_height, self.input_width])
        # label = tf.Print(label, [label[label.shape[0] - 1, 1]], 'label')
        # label = tf.Print(label, [label[tf.shape(label)[0] - 1, 1]], 'label')
        image = tf.py_func(self.pyfunc_resize, [image, label], (tf.float32))
        image.set_shape((self.input_height, self.input_width, 3))


        # print(image.dtype)
        # print(image.shape)
        return image, label

    def pyfunc_resize(self, image, label):
        # print('pyfunc_resize')
        # print(label)
        image = image.astype(np.uint8)
        # print(image.__class__.__name__)
        # print(image.dtype)
        # print(image.shape)
        # print(image[9, 0, 0], image[10, 0, 0], image[11, 0, 0])
        # print(image[9, 1, 0], image[10, 1, 0], image[11, 1, 0])
        # print(image[9, 0, 1], image[10, 0, 1], image[11, 0, 1])
        # print(image[9, 1, 1], image[10, 1, 1], image[11, 1, 1])
        # print(image[9, 0, 2], image[10, 0, 2], image[11, 0, 2])
        # print(image[9, 1, 2], image[10, 1, 2], image[11, 1, 2])
        image = cv2.resize(image, (self.input_height, self.input_width), interpolation=1)
        # image = np.round(image)
        image = image.astype(np.float32)
        # print('after reshape')
        # print(image.__class__.__name__)
        # print(image.dtype)
        # print(image.shape)
        # print(image[9, 0, 0], image[10, 0, 0], image[11, 0, 0])
        # print(image[9, 1, 0], image[10, 1, 0], image[11, 1, 0])
        # print(image[9, 0, 1], image[10, 0, 1], image[11, 0, 1])
        # print(image[9, 1, 1], image[10, 1, 1], image[11, 1, 1])
        # print(image[9, 0, 2], image[10, 0, 2], image[11, 0, 2])
        # print(image[9, 1, 2], image[10, 1, 2], image[11, 1, 2])
        # print('pyfunc_resize listo')
        return image

    def resize_pad_zeros(self, image, bboxes):
        # Resize image so the biggest side fits exactly in the input size:
        # height, width = tf.shape(image)[0], tf.shape(image)[1]
        width, height = tf.shape(image)[0], tf.shape(image)[1]
        scale_height = self.input_height / tf.to_float(height)
        scale_width = self.input_width / tf.to_float(width)
        scale = tf.minimum(scale_height, scale_width)
        tf.cast(tf.round(scale * tf.to_float(height)), tf.int32)
        new_height = tf.minimum(tf.cast(tf.round(scale * tf.to_float(height)), tf.int32), self.input_height)
        new_width = tf.minimum(tf.cast(tf.round(scale * tf.to_float(width)), tf.int32), self.input_width)
        # size = tf.stack([new_height, new_width])
        size = tf.stack([new_width, new_height])
        image = tf.image.resize_images(image, size)
        # Pad the image with zeros and modify accordingly the bounding boxes:
        (image, bboxes) = tf.py_func(self.pad_with_zeros, (image, size, bboxes), (tf.float32, tf.float32))
        image.set_shape((self.input_height, self.input_width, 3))
        bboxes.set_shape((None, 5))
        return image, bboxes

    def pad_with_zeros(self, image, size, bboxes):
        # Increment on each side:
        increment_height = int(self.input_height - size[0])
        increment_top = int(np.round(increment_height / 2.0))
        increment_bottom = increment_height - increment_top
        increment_width = int(self.input_width - size[1])
        increment_left = int(np.round(increment_width / 2.0))
        increment_right = increment_width - increment_left
        image = cv2.copyMakeBorder(image, increment_top, increment_bottom, increment_left, increment_right, cv2.BORDER_CONSTANT)
        # Warp and shift boxes:
        rel_incr_left = float(increment_left) / size[1]
        rel_incr_right = float(increment_right) / size[1]
        rel_incr_top = float(increment_top) / size[0]
        rel_incr_bottom = float(increment_bottom) / size[0]
        for i in range(len(bboxes)):
            bboxes[i][1] = (bboxes[i][1] + rel_incr_left) / (1.0 + rel_incr_left + rel_incr_right)
            bboxes[i][2] = (bboxes[i][2] + rel_incr_top) / (1.0 + rel_incr_top + rel_incr_bottom)
            bboxes[i][3] = bboxes[i][3] / (1.0 + rel_incr_left + rel_incr_right)
            bboxes[i][4] = bboxes[i][4] / (1.0 + rel_incr_top + rel_incr_bottom)
        return image, bboxes


def ResizeNumpy(image, method, input_width, input_height):
    if method == 'resize_warp':
        image = cv2.resize(image, (input_width, input_height))
    elif method == 'resize_pad_zeros':
        height, width, _ = image.shape
        # Resize so it fits totally in the input size:
        scale_width = input_width / np.float32(width)
        scale_height = input_height / np.float32(height)
        scale = min(scale_width, scale_height)
        new_width = int(np.round(width * scale))
        new_height = int(np.round(height * scale))
        image = cv2.resize(image, (new_width, new_height))
        # Pad with zeros the remaining areas:
        increment_height = int(input_height - new_height)
        increment_top = int(np.round(increment_height / 2.0))
        increment_bottom = increment_height - increment_top
        increment_width = int(input_width - new_width)
        increment_left = int(np.round(increment_width / 2.0))
        increment_right = increment_width - increment_left
        image = cv2.copyMakeBorder(image, increment_top, increment_bottom, increment_left, increment_right,
                                   cv2.BORDER_CONSTANT)
    elif method == 'resize_lose_part':
        raise Exception('resize_lose_part not implemented for InteractiveDataReader')
    elif method == 'centered_crop':
        raise Exception('centered_crop not implemented for InteractiveDataReader')
    else:
        raise Exception('Resize method not recognized.')
    return image

