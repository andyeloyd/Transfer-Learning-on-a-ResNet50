import tensorflow as tf
import math


# https://github.com/peteryuX/arcface-tf2/blob/master/modules/layers.py
# Slightly modified version

class ArcFace(tf.keras.layers.Layer):
    def __init__(self, n_classes=10, s=30.0, m=0.50, regularizer=None, **kwargs):
        super(ArcFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = regularizer

    def get_config(self):
        config = super().get_config()
        config.update({
            "n_classes": self.n_classes,
            "logits_scale": self.s,
            "margin": self.m,
            "regularizer": self.regularizer
        })
        return config

    def build(self, input_shape):
        # self.w = self.add_variable(
        #    "weights", shape=[int(input_shape[-1]), self.n_classes])
        self.w = self.add_weight(
            name='weights', shape=(int(input_shape[-1]), self.n_classes),
            initializer='glorot_uniform', regularizer=self.regularizer, trainable=True)
        self.cos_m = tf.identity(math.cos(self.m), name='cos_m')
        self.sin_m = tf.identity(math.sin(self.m), name='sin_m')
        self.th = tf.identity(math.cos(math.pi - self.m), name='th')
        self.mm = tf.multiply(self.sin_m, self.m, name='mm')

    def call(self, embds, labels):
        normed_embds = tf.nn.l2_normalize(embds, axis=1, name='normed_embd')
        normed_w = tf.nn.l2_normalize(self.w, axis=0, name='normed_weights')

        cos_t = tf.matmul(normed_embds, normed_w, name='cos_t')
        sin_t = tf.sqrt(1. - cos_t ** 2, name='sin_t')

        cos_mt = tf.subtract(
            cos_t * self.cos_m, sin_t * self.sin_m, name='cos_mt')

        cos_mt = tf.where(cos_t > self.th, cos_mt, cos_t - self.mm)

        mask = tf.one_hot(tf.cast(labels, tf.int32), depth=self.n_classes,
                          name='one_hot_mask')

        logits = tf.where(mask == 1., cos_mt, cos_t)
        logits = tf.multiply(logits, self.s, 'arcface_logits')

        #return logits
        return tf.nn.softmax(logits)