import tensorflow as tf
import numpy as np


def batch_entropy(x):
    """ Estimate the differential entropy by assuming a gaussian distribution of
        values for different samples of a mini-batch.
    """
    if x.shape[0] <= 1:
        raise Exception("The batch entropy can only be calculated for |batch| > 1.")

    x = tf.reshape(x, [tf.shape(x)[0], -1])
    x_std = tf.math.reduce_std(x, axis=0)
    entropies = 0.5 * tf.math.log(np.pi * np.e * x_std ** 2 + 1e-10)  # Added epsilon to prevent log(0)
    return tf.reduce_mean(entropies)


class LBELoss(tf.keras.losses.Loss):
    def __init__(self, num_layers, lbe_alpha=0.5, lbe_alpha_min=0.5, lbe_beta=0.5, **kwargs):
        super(LBELoss, self).__init__(**kwargs)
        self.ce = tf.keras.losses.SparseCategoricalCrossentropy()
        self.lbe_alpha_p = tf.Variable(tf.ones([num_layers]) * lbe_alpha, trainable=True)
        self.lbe_alpha_min = tf.convert_to_tensor(lbe_alpha_min)
        self.lbe_beta = lbe_beta

    def lbe_per_layer(self, a, i):
        lbe_alpha_l = tf.abs(self.lbe_alpha_p[i])
        lbe_l = (batch_entropy(a) - tf.maximum(self.lbe_alpha_min, lbe_alpha_l)) ** 2
        return lbe_l * self.lbe_beta

    def call(self, y_true, y_pred):
        output, A = y_pred
        ce = self.ce(y_true, output)

        if A is None:
            print("no network A passed; returning empty array")
            return []

        losses = [self.lbe_per_layer(a, i) for i, a in enumerate(A)]
        #lbe = tf.reduce_mean(losses) * ce
        #return ce + lbe, ce, lbe
        return losses # we need layerwise batch entropy instead of a single metric for the entire network


class CELoss(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super(CELoss, self).__init__(**kwargs)
        self.ce = tf.keras.losses.SparseCategoricalCrossentropy()

    def call(self, y_true, y_pred):
        output, _ = y_pred
        ce = self.ce(y_true, output)
        return ce, ce, 0.0