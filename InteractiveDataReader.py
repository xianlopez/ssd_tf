import tensorflow as tf
import cv2
import numpy as np
import Resizer


class InteractiveDataReader:

    def __init__(self, input_side):
        self.input_side = input_side
        self.mean = [123.68, 116.78, 103.94]

    def build_inputs(self):
        return tf.placeholder(dtype=tf.float32, shape=(None, self.input_side, self.input_side, 3), name='inputs')

    def get_batch(self, image_paths):
        batch_size = len(image_paths)
        inputs_numpy = np.zeros(shape=(batch_size, self.input_side, self.input_side, 3), dtype=np.float32)
        for i in range(batch_size):
            inputs_numpy[i, :, :, :] = self.get_image(image_paths[i])
        return inputs_numpy

    def preprocess_image(self, image):
        means = np.zeros(shape=image.shape, dtype=np.float32)
        for i in range(3):
            means[:, :, i] = self.mean[i]
        image = image - means
        # Resize:
        # image = Resizer.ResizeNumpy(image, 'resize_pad_zeros', self.input_side, self.input_side)
        image = Resizer.ResizeNumpy(image, 'resize_warp', self.input_side, self.input_side)
        return image

    def get_image(self, image_path):
        # Read image:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # Preprocess it:
        image = self.preprocess_image(image)
        return image