import tensorflow as tf

class PreprocessOpts:
    type = 'subtract_mean'  # 'fit_to_range', 'subtract_mean'
    range_min = 0
    range_max = 0.4366
    mean = 'vgg'

# VGG_MEAN = [123.68, 116.78, 103.94]
VGG_MEAN = [123.0, 117.0, 104.0]


class Preprocessor:

    def __init__(self, preprocess_opts, width, height):
        self.preprocess_opts = preprocess_opts
        self.width = width
        self.height = height
        self.mean = None
        self.range_min = None
        self.range_max = None

    def get_preprocess_function(self):
        if self.preprocess_opts.type == 'fit_to_range':
            self.range_min = self.preprocess_opts.range_min
            self.range_max = self.preprocess_opts.range_max
            preprocess_function = self.fit_to_range
        elif self.preprocess_opts.type == 'subtract_mean':
            if self.preprocess_opts.mean == 'vgg':
                self.mean = VGG_MEAN
            else:
                raise Exception('Preprocess mean not recognized.')
            preprocess_function = self.subtract_mean
        else:
            raise Exception('Preprocess type not recognized.')
        return preprocess_function

    def fit_to_range(self, image):
        image = self.range_min + image * (self.range_max - self.range_min) / 255.0
        return image

    def subtract_mean(self, image):
        means = tf.reshape(tf.constant(self.mean), [1, 1, 3])
        image = image - means
        return image