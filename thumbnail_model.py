import numpy as np
import tensorflow as tf

class ThumbnailModel(tf.keras.Model):

    def __init__(self, hidden_size, filter_size, embed_size, vocab_size, max_win):
        super().__init__()
        self.hidden_size = hidden_size
        self.filter_num = filter_size
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.max_win = max_win
        self.optimizer = tf.keras.optimizers.Adam()
        self.image_arch = tf.keras.Sequential(
            layers = [
                # tf.keras.layers.Reshape((120, 90, 3)),
                tf.keras.layers.Conv2D(self.filter_num, 4, padding = "SAME"),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.MaxPool2D(padding = "SAME", pool_size=(3, 3), strides=2),


                tf.keras.layers.Conv2D(5, 1, padding = "SAME"),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.MaxPool2D(padding = "SAME", pool_size=(3, 3), strides=2),

                tf.keras.layers.Conv2D(3, 1, padding = "SAME"),
                tf.keras.layers.LeakyReLU(),

                tf.keras.layers.Conv2D(3, 1, padding = "SAME"),
                tf.keras.layers.LeakyReLU(),

                tf.keras.layers.Conv2D(1, 1, padding = "SAME"),
                tf.keras.layers.LeakyReLU(),
                
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Flatten()
            ])
        self.text_arch = tf.keras.Sequential(
            layers = [
                tf.keras.layers.Conv1D(self.filter_num, 3, padding = "SAME"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.MaxPool1D(padding = "SAME"),
                tf.keras.layers.Conv1D(1, 3, padding = "SAME"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Flatten(),
            ])
        self.numerical_arch = tf.keras.Sequential(
            layers = [
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU()
            ])
        self.feed_forward = tf.keras.Sequential(
            layers = [
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(self.hidden_size),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dense(1),
                # tf.keras.layers.ReLU()
            ])
        self.embed_layer = tf.keras.layers.Embedding(input_dim = self.vocab_size, output_dim=self.embed_size, trainable = True, input_length=self.max_win)

    # @tf.function #NOT SURE WHAT THE FUCK THIS IS FOR
    def call(self, images, texts, numbers):
        # images =tf.cast(images, tf.int32)
        out1 = self.image_arch(images)
        text_vecs = self.embed_layer(texts)
        out2 = tf.reshape(self.text_arch(text_vecs), [texts.shape[0],-1])
        out3 = self.numerical_arch(numbers)
        in4 = tf.concat([out1, out2, out3], axis = 1)
        estimate = self.feed_forward(in4)
        return  estimate
    ##NO IDEA WHAT THESE METHDOS EVEN DO. THEY SEEM FUCKED UP
    # def get_config(self):
    #     return {"decoder": self.decoder} ## specific to ImageCaptionModel

    # @classmethod
    # def from_config(cls, config):
    #     return cls(**config)


    def compile(self, optimizer, loss, metrics):
        '''
        Create a facade to mimic normal keras fit routine
        '''
        self.optimizer = optimizer
        self.loss_function = loss 
        self.accuracy_function = metrics[0]


def accuracy_function(preds, labels):
    return tf.reduce_mean(tf.keras.metrics.mean_absolute_percentage_error(labels, preds))


def loss_function(preds, labels):
    return tf.reduce_mean(tf.keras.metrics.mean_squared_error(labels, preds))