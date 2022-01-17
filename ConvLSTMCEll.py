import tensorflow as tf
from tensorflow import keras

class ConvLSTMCell(tf.keras.Model):
    def __init__(self, hidden_dim,att_hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.bias = bias
        self.attention_layer = self_attention_memory_module(hidden_dim,att_hidden_dim)

        self.conv = tf.keras.layers.Conv2D(
            filters = 4 * self.hidden_dim,
            kernel_size = self.kernel_size,
            padding = 'same',
            use_bias = self.bias
        )

    def call(self, input_tensor, cur_state):
        h_cur, c_cur, m_cur = cur_state
        combined = tf.concat([input_tensor, h_cur], axis=1)
        combined_conv = self.conv(combined)
        # num_or_size_splits 이거 self.hidden_dim으로 바꿀수도   원래는 axis=-1 , num_or = 4였음
        cc_i, cc_f, cc_o, cc_g = tf.split(combined_conv, num_or_size_splits=self.hidden_dim, axis=1)
        i = tf.keras.activations.sigmoid(cc_i)
        f = tf.keras.activations.sigmoid(cc_f)
        o = tf.keras.activations.sigmoid(cc_o)
        g = tf.keras.activations.tanh(cc_g)

        c_next = f*c_cur+i*g
        h_next = o*tf.keras.activations.tanh(c_next)

        # attention
        h_next, m_next = self.attention_layer(h_next,m_cur)

        return h_next, (h_next,c_next,m_next)

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (tf.zeros([batch_size, self.hidden_dim, height, width]),
                tf.zeros([batch_size, self.hidden_dim, height, width]),
                tf.zeros([batch_size, self.hidden_dim, height, width])
                )