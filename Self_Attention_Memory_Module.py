import tensorflow as tf
class Self_Attention_Memory_Module(tf.keras.Model):
    def __init__(self, input_dim,hidden_dim, kernel_size, bias):
        super(Self_Attention_Memory_Module, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.layer_q = tf.keras.layers.Conv2D(input_dim, hidden_dim, kernel_size)
        self.layer_k = tf.keras.layers.Conv2D(input_dim, hidden_dim, kernel_size)
        self.layer_k2 = tf.keras.layers.Conv2D(input_dim, hidden_dim, kernel_size)
        self.layer_v = tf.keras.layers.Conv2D(input_dim, input_dim, kernel_size)
        self.layer_v2 = tf.keras.layers.Conv2D(input_dim, input_dim, kernel_size)
        self.layer_z = tf.keras.layers.Conv2D(input_dim * 2, input_dim * 2, kernel_size)
        self.layer_m = tf.keras.layers.Conv2D(input_dim * 3, input_dim * 3, kernel_size)


    def call(self,h,m):
        batch_size, channel, H, W = h.shape
        # feature aggregation
        # hidden h attention
        k_h = self.layer_k(h)
        q_h = self.layer_q(h)
        k_h = k_h.reshape(batch_size, self.hidden_dim, H * W)
        q_h = q_h.reshape(batch_size, self.hidden_dim, H * W)
        q_h = tf.transpose(q_h , perm=[1, 2])
        a_h = tf.nn.softmax(tf.matmul(q_h, k_h), axis=-1)  # batch_size, H*W, H*W
        v_h = self.layer_v(h)
        v_h = v_h.reshape(batch_size, self.input_dim, H * W)
        z_h = tf.matmul(a_h, tf.tranpose(v_h, perm=[0, 2, 1]))

        # memory m attention
        k_m = self.layer_k2(m)
        v_m = self.layer_v2(m)
        k_m = k_m.reshape(batch_size, self.hidden_dim, H * W)
        v_m = v_m.reshape(batch_size, self.input_dim, H * W)
        a_m = tf.nn.softmax(tf.matmul(q_h, k_m), axis=-1)
        v_m = self.layer_v2(m)
        v_m = v_m.reshape(batch_size, self.input_dim, H * W)
        z_m = tf.matmul(a_m, tf.tranpose(v_m, perm=[0, 2, 1]))
        z_h = tf.tranpose(z_h,perm=[1,2]).reshape(batch_size, self.input_dim, H, W)
        z_m = tf.tranpose(z_m,perm=[1,2]).reshape(batch_size, self.input_dim, H, W)

        w_z = tf.concat([z_h, z_m], 1)
        Z = self.layer_z(w_z)
        # Memory Updating
        combined = self.layer_m(tf.concat([Z, h], 1))  # 3 * input_dim
        mo, mg, mi = tf.split(combined, self.input_dim, axis=1)

        mi = tf.keras.activations.sigmoid(mi)
        new_m = (1 - mi) * m + mi * tf.keras.activations.tanh(mg)
        new_h = tf.keras.activations.sigmoid(mo) * new_m

        return new_h, new_m

