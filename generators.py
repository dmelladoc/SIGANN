import numpy as np
import tensorflow as tf
from net_utils import one_hot

class InputGenerator:
    def __init__(self, input_shape, n_classes, z_size=10, batch_size=32, n_epochs=10):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.z_size = z_size
        self.batch_size = batch_size
        self.n_epochs = n_epochs

    def RNG(self):
        while True:
            z = np.random.uniform(-np.sqrt(3), np.sqrt(3), self.z_size)
            y = np.random.randint(0, self.n_classes)
            y = one_hot(y, self.n_classes).squeeze()
            yield (z, y)

    def train_augment(self, x, y):
        shape = tf.shape(x)
        img = tf.image.resize_image_with_crop_or_pad(x, 33, 33)
        img = tf.random_crop(img, shape)
        img = tf.image.random_brightness(img, 0.15)
        img = tf.cast(tf.greater_equal(img, 0.5), tf.float32)
        return img, y

    def eval_augment(self, x, y):
        img = tf.cast(tf.greater_equal(x, 0.5), tf.float32)
        return img, y

    def test_augment(self, x):
        img = tf.cast(tf.greater_equal(x, 0.5), tf.float32)
        return img

    def create_train_generator(self):
        self.x_input = tf.placeholder(tf.float32, self.input_shape, name="X_input")
        self.y_input = tf.placeholder(tf.float32, [None, self.n_classes], name="Y_input")

        #Generadores:
        d1 = tf.data.Dataset.from_tensor_slices((self.x_input, self.y_input)).map(self.train_augment)
        d2 = tf.data.Dataset.from_generator(self.RNG, (tf.float32, tf.float32), (tf.TensorShape([self.z_size]), tf.TensorShape([self.n_classes]))).prefetch(5000)

        df = tf.data.Dataset.zip((d1,d2)).batch(self.batch_size).apply(tf.contrib.data.shuffle_and_repeat(5000, self.n_epochs))
        return df.make_initializable_iterator()

    def create_eval_generator(self):
        self.x_input = tf.placeholder(tf.float32, self.input_shape, name="X_input")
        self.y_input = tf.placeholder(tf.float32, [None, self.n_classes], name="Y_input")

        df = tf.data.Dataset.from_tensor_slices((self.x_input, self.y_input)).apply(tf.contrib.data.map_and_batch(self.eval_augment, self.batch_size))
        return df.make_initializable_iterator()

    def create_gen_generator(self):
        df = tf.data.Dataset.from_generator(self.RNG, (tf.float32, tf.float32), (tf.TensorShape([self.z_size]), tf.TensorShape([self.n_classes]))).prefetch(5000).batch(self.batch_size)
        return df.make_initializable_iterator()

    def create_test_generator(self):
        self.x_input = tf.placeholder(tf.float32, self.input_shape, name="X_input")
        self.y_input = None
        df = tf.data.Dataset.from_tensor_slices(self.x_input).apply(tf.contrib.data.map_and_batch(self.test_augment, self.batch_size))
        return df.make_initializable_iterator()
