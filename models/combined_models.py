import tensorflow as tf

SEED = 0

class ImageNumModel(tf.keras.Model):

    def __init__(self, hidden_size, filter_size, embed_size, vocab_size, window_size, desired_learning_rate):
        super().__init__()
        self.hidden_size = hidden_size
        self.filter_size = filter_size
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.window_size = window_size

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=desired_learning_rate)

        # inspired by http://cs231n.stanford.edu/reports/2017/pdfs/710.pdf
        self.images_arch = tf.keras.Sequential(
            layers=[
                tf.keras.layers.Conv2D(self.filter_size, 4, padding="SAME", strides=4, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.MaxPool2D(padding="SAME", pool_size=(3, 3), strides=2),

                tf.keras.layers.Conv2D(5, 1, padding="SAME", kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED)),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.MaxPool2D(padding="SAME", pool_size=(3, 3), strides=2),

                tf.keras.layers.Conv2D(3, 1, padding="SAME", kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED)),
                tf.keras.layers.LeakyReLU(),

                tf.keras.layers.Conv2D(3, 1, padding="SAME", kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED)),
                tf.keras.layers.LeakyReLU(),

                tf.keras.layers.Conv2D(1, 1, padding="SAME", kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED)),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.MaxPool2D(padding="SAME", pool_size=(3, 3), strides=2),

                tf.keras.layers.Dropout(0.5),

                tf.keras.layers.Flatten()
            ]
        )

        self.nums_arch = tf.keras.Sequential(
            layers=[
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(2, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED)),
                tf.keras.layers.LeakyReLU()
            ]
        )

        self.feed_forward = tf.keras.Sequential(
            layers=[
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(self.hidden_size, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED)),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dropout(0.5),

                tf.keras.layers.Dense(self.hidden_size / 2, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED)),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dropout(0.5),

                tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED)),
            ]
        )

    def call(self, images, text, nums):
        images_output = self.images_arch(images)
        nums_output = self.nums_arch(nums)

        combined_output = tf.concat([images_output, nums_output], axis=1)
        estimation = self.feed_forward(combined_output)

        return estimation

    def compile(self, optimizer, loss, metrics):
        self.optimizer = optimizer
        self.loss_function = loss
        self.accuracy_function = metrics[0]


class ImageTextModel(tf.keras.Model):

    def __init__(self, hidden_size, filter_size, embed_size, vocab_size, window_size, desired_learning_rate):
        super().__init__()
        self.hidden_size = hidden_size
        self.filter_size = filter_size
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.window_size = window_size

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=desired_learning_rate)
        self.embed_layer = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embed_size, trainable=True,
                                                    embeddings_initializer=tf.keras.initializers.RandomUniform(seed=SEED), input_length=self.window_size)

        # inspired by http://cs231n.stanford.edu/reports/2017/pdfs/710.pdf
        self.images_arch = tf.keras.Sequential(
            layers=[
                tf.keras.layers.Conv2D(self.filter_size, 4, padding="SAME", strides=4, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.MaxPool2D(padding="SAME", pool_size=(3, 3), strides=2),

                tf.keras.layers.Conv2D(5, 1, padding="SAME", kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED)),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.MaxPool2D(padding="SAME", pool_size=(3, 3), strides=2),

                tf.keras.layers.Conv2D(3, 1, padding="SAME", kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED)),
                tf.keras.layers.LeakyReLU(),

                tf.keras.layers.Conv2D(3, 1, padding="SAME", kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED)),
                tf.keras.layers.LeakyReLU(),

                tf.keras.layers.Conv2D(1, 1, padding="SAME", kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED)),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.MaxPool2D(padding="SAME", pool_size=(3, 3), strides=2),

                tf.keras.layers.Dropout(0.5),

                tf.keras.layers.Flatten()
            ]
        )

        self.text_arch = tf.keras.Sequential(
            layers=[
                tf.keras.layers.Conv1D(self.filter_size, 3, padding="SAME", kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.MaxPool1D(padding="SAME"),

                tf.keras.layers.Conv1D(1, 3, padding="SAME", kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(),

                tf.keras.layers.Flatten(),
            ]
        )

        self.feed_forward = tf.keras.Sequential(
            layers=[
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(self.hidden_size, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED)),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dropout(0.5),

                tf.keras.layers.Dense(self.hidden_size / 2, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED)),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dropout(0.5),

                tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED)),
            ]
        )

    def call(self, images, text, nums):
        images_output = self.images_arch(images)
        text_output = tf.reshape(self.text_arch(self.embed_layer(text)), [text.shape[0], -1])

        combined_output = tf.concat([images_output, text_output], axis=1)
        estimation = self.feed_forward(combined_output)

        return estimation

    def compile(self, optimizer, loss, metrics):
        self.optimizer = optimizer
        self.loss_function = loss
        self.accuracy_function = metrics[0]


class NumTextModel(tf.keras.Model):

    def __init__(self, hidden_size, filter_size, embed_size, vocab_size, window_size, desired_learning_rate):
        super().__init__()
        self.hidden_size = hidden_size
        self.filter_size = filter_size
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.window_size = window_size

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=desired_learning_rate)
        self.embed_layer = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embed_size, trainable=True,
                                                    embeddings_initializer=tf.keras.initializers.RandomUniform(seed=SEED), input_length=self.window_size)

        self.text_arch = tf.keras.Sequential(
            layers=[
                tf.keras.layers.Conv1D(self.filter_size, 3, padding="SAME", kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.MaxPool1D(padding="SAME"),

                tf.keras.layers.Conv1D(1, 3, padding="SAME", kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(),

                tf.keras.layers.Flatten(),
            ]
        )

        self.nums_arch = tf.keras.Sequential(
            layers=[
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(2, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED)),
                tf.keras.layers.LeakyReLU()
            ]
        )

        self.feed_forward = tf.keras.Sequential(
            layers=[
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(self.hidden_size, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED)),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dropout(0.5),

                tf.keras.layers.Dense(self.hidden_size / 2, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED)),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dropout(0.5),

                tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED)),
            ]
        )

    def call(self, images, text, nums):
        text_output = tf.reshape(self.text_arch(self.embed_layer(text)), [text.shape[0], -1])
        nums_output = self.nums_arch(nums)

        combined_output = tf.concat([text_output, nums_output], axis=1)
        estimation = self.feed_forward(combined_output)

        return estimation

    def compile(self, optimizer, loss, metrics):
        self.optimizer = optimizer
        self.loss_function = loss
        self.accuracy_function = metrics[0]