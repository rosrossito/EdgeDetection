from keras.layers import Layer
from keras import backend as K

FIRING_THRESHOLD = 0.85

class Custom_relu(Layer):
    def __init__(self, shape, **kwargs):
        super(Custom_relu, self).__init__(**kwargs)
        self.shape = shape

    def call(self, inputs, **kwargs):
        return K.relu(K.abs(inputs)/self.shape, alpha=0., max_value=1, threshold=FIRING_THRESHOLD)
        # return K.relu(K.abs(inputs)/self.shape)
