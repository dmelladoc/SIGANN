import tensorflow as tf
import numpy as np


class AAE:
    def __init__(self, stage, batch_size = 32, n_epochs=10, n_classes=10, z_size=16, input_shape=[None, 28, 28, 1]):
        self.training = True if stage == "train" else False
        self.n_epochs = n_epochs
        self.n_classes = n_classes
        self.shape = input_shape
        self.batch_size = batch_size
        self.z_size = z_size
        self.quarter_shape = [-1] + [s // 4 for s in input_shape[1:-1]] + [128]
        self.quarter_flat = np.prod(self.quarter_shape[1:])
        self.epsilon = tf.constant(1e-8)

    def _conv_layer(self, x, n_neurons, kernel_size=3, activation=None, strides=1, name="ConvLayer"):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            conv = tf.layers.conv2d(x, n_neurons, kernel_size=kernel_size, kernel_initializer=tf.variance_scaling_initializer(distribution="uniform"), strides=strides, padding="same", name="conv")
            BN = tf.layers.batch_normalization(conv, training=self.training, fused=True, name="BN")
            return activation(BN) if activation is not None else BN

    def resnet_layer(self, x, n_neurons, kind="conv", name="resnet_layer"):
        if kind == "conv":
            layer = self._conv_layer
        else:
            layer = self._deconv_layer
        neurons = n_neurons // 8
        with tf.variable_scope(name, initializer=tf.variance_scaling_initializer(distribution="uniform"), reuse=tf.AUTO_REUSE):
            stack_1A = layer(x, neurons*2, kernel_size=1, activation=tf.nn.relu, name="Stack1")

            stack_2A = layer(x, neurons, kernel_size=1, name="Stack2A")
            stack_2B = layer(stack_2A, neurons*2, kernel_size=3, activation=tf.nn.relu, name="Stack2B")

            stack_3A = layer(x, neurons, kernel_size=1, name="Stack3A")
            stack_3B = layer(stack_3A, neurons*2, kernel_size=3, activation=tf.nn.relu, name="Stack3B")
            stack_3C = layer(stack_3B, neurons*2, kernel_size=3, activation=tf.nn.relu, name="Stack3C")

            stack_4A = tf.layers.max_pooling2d(x, 3, strides=1, padding="same", name="Stack4A")
            stack_4B = layer(stack_4A, neurons*2, kernel_size=1, activation=tf.nn.relu, name="Stack4B")

            return tf.concat([stack_1A, stack_2B, stack_3C, stack_4B], -1, name="concat")

    def _deconv_layer(self, x, n_neurons, kernel_size=3, activation=None, strides=1, name="DeConvLayer"):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            conv = tf.layers.conv2d_transpose(x, n_neurons, kernel_size=kernel_size, kernel_initializer=tf.variance_scaling_initializer(distribution="uniform"), strides=strides, padding="same", name="deconv")
            BN = tf.layers.batch_normalization(conv, training=self.training, fused=True, name="BN")
            return activation(BN) if activation is not None else BN

    def encoder(self, x, supervised=False):
        with tf.variable_scope("Encoder", reuse=tf.AUTO_REUSE):
            layer_1 = self._conv_layer(x, 32, kernel_size=3, activation=tf.nn.relu, strides=1, name="Layer1")
            layer_2 = self.resnet_layer(layer_1, 32, kind="conv", name="Layer2")

            layer_3 = self._conv_layer(layer_2, 64, kernel_size=3, activation=tf.nn.relu, strides=2, name="Layer3")
            layer_4 = self.resnet_layer(layer_3, 64, kind="conv", name="Layer4")

            layer_5 = self._conv_layer(layer_4, 128, kernel_size=3, activation=tf.nn.relu, strides=2, name="Layer5")
            layer_6 = self.resnet_layer(layer_5, 128, kind="conv", name="Layer6")
            flatten = tf.layers.flatten(layer_6, name="Flatten")

            fc_z = tf.layers.dense(flatten, 512, kernel_initializer=tf.variance_scaling_initializer(distribution="uniform"), activation=tf.nn.relu, name="fcB")
            mu = tf.layers.dense(fc_z, self.z_size, kernel_initializer=tf.variance_scaling_initializer(distribution="uniform"), name="Z")

            fc_a = tf.layers.dense(flatten, 512, kernel_initializer=tf.variance_scaling_initializer(distribution="uniform"), activation=tf.nn.relu, name="fcA")
            y_hat = tf.layers.dense(fc_a, self.n_classes, kernel_initializer=tf.variance_scaling_initializer(distribution="uniform"), name="Y_logits")  # Ver como afecta el ponerle detención de gradiente aqui
            if not supervised:
                return mu, tf.nn.softmax(y_hat, name="Yhat")
            else:
                return mu, y_hat

    def decoder(self, z, y):
        with tf.variable_scope("Decoder", reuse=tf.AUTO_REUSE):
            cat = tf.concat([z, y], axis=-1, name="Concatenate")
            fc1 = tf.layers.dense(cat, self.quarter_flat, kernel_initializer=tf.variance_scaling_initializer(distribution="uniform"), activation=tf.nn.relu, name="Y_fc_dec")  # 7x7x128 porque mnist :v
            to_image = tf.reshape(fc1, self.quarter_shape, name="reshape")
            layer_8 = self.resnet_layer(to_image, 128, kind="deconv", name="Layer8")
            layer_9 = self._deconv_layer(layer_8, 128, 3, activation=tf.nn.relu, strides=2, name="Layer9")

            layer_A = self.resnet_layer(layer_9, 64, kind="deconv", name="LayerA")
            layer_B = self._deconv_layer(layer_A, 64, 3, activation=tf.nn.relu, strides=2, name="LayerB")

            layer_C = self.resnet_layer(layer_B, 32, kind="deconv", name="LayerC")
            x_recon = tf.layers.conv2d_transpose(layer_C, self.shape[-1], kernel_size=3, kernel_initializer=tf.variance_scaling_initializer(distribution="uniform"), padding="same", activation=tf.nn.sigmoid, name="Xrecon")
            return x_recon

    def binary_crossentropy(self, real, predicho):
        with tf.name_scope("binary_crossentropy"):
            true = tf.layers.flatten(real)
            pred = tf.layers.flatten(predicho)
            CE = -tf.reduce_sum((true * tf.log(pred + self.epsilon)) + ((1.0 - true) * tf.log(1.0 - pred + self.epsilon)), 1)
            return CE

    def discriminator_z(self, z):
        with tf.variable_scope("Discriminator_Z", reuse=tf.AUTO_REUSE):
            dz_1 = tf.layers.dense(z, 512, activation=tf.nn.relu, kernel_initializer=tf.variance_scaling_initializer(distribution="uniform"), name="dz_l1")
            dz_2 = tf.layers.dense(dz_1, 512, activation=tf.nn.relu, kernel_initializer=tf.variance_scaling_initializer(distribution="uniform"), name="dz_l2")
            output = tf.layers.dense(dz_2, 1, kernel_initializer=tf.variance_scaling_initializer(distribution="uniform"), name="dz_out")
            return output

    def discriminator_y(self, y):
        with tf.variable_scope("Discriminator_Y", reuse=tf.AUTO_REUSE):
            dy_1 = tf.layers.dense(y, 512, activation=tf.nn.relu, kernel_initializer=tf.variance_scaling_initializer(distribution="uniform"), name="Y_dy_l1")
            dy_2 = tf.layers.dense(dy_1, 512, activation=tf.nn.relu, kernel_initializer=tf.variance_scaling_initializer(distribution="uniform"), name="dy_l2")
            output = tf.layers.dense(dy_2, 1, kernel_initializer=tf.variance_scaling_initializer(distribution="uniform"), name="dy_out")
            return output
