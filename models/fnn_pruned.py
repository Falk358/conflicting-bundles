import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa


class FNN_pruned(tf.keras.Model):
    
    def __init__(self, config, layer_count):
        super(FNN_pruned, self).__init__()
        self.cb = None
        self.fc_layers = []
        self.config = config
        self.image_dim = self.config.img_width * self.config.img_height * self.config.img_depth

        for i in range(layer_count-1):
            self.fc_layers.append(
                layers.Dense(config.width_layers, 
                name="fc%d" % i, 
                activation="relu",
                kernel_initializer="he_uniform",
                bias_initializer=tf.zeros_initializer)
            )

        self.out = layers.Dense(config.num_classes, 
                name="out", 
                activation="linear")


    def call(self, x, training=True):
        self.cb = []
        batch_size = tf.shape(x)[0]

        # Reshape input images
        x = tf.reshape(x, [batch_size, -1])

        # FC hidden layer
        for l in range(len(self.fc_layers)):
            x = self.fc_layers[l](x)
            self.cb.append((x, l))

        # Output layer
        x = self.out(x)
        return x