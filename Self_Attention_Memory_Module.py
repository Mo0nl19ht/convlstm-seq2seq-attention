import tensorflow as tf
class Self_Attention_Memory_Module(tf.keras.Model):
    #파라미터에서 input_dim지우기
    def __init__(self, input_dim,hidden_dim, kernel_size):
        super(Self_Attention_Memory_Module, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.layer_q = tf.keras.layers.Conv2D(hidden_dim, kernel_size,padding="same")
        self.layer_k = tf.keras.layers.Conv2D(hidden_dim, kernel_size,padding="same")
        self.layer_k2 = tf.keras.layers.Conv2D(hidden_dim, kernel_size,padding="same")
        self.layer_v = tf.keras.layers.Conv2D(hidden_dim, kernel_size,padding="same")
        self.layer_v2 = tf.keras.layers.Conv2D(hidden_dim, kernel_size,padding="same")
        self.layer_z = tf.keras.layers.Conv2D(hidden_dim * 2, kernel_size,padding="same")
        self.layer_m = tf.keras.layers.Conv2D(hidden_dim * 3, kernel_size,padding="same")


    def call(self,h,m):
        batch_size, H, W,channel = h.shape
        # feature aggregation
        # hidden h attention
        # print(h.shape,"asd")
        k_h = self.layer_k(h)
        q_h = self.layer_q(h)
        # print(k_h.shape)
        k_h = tf.reshape(k_h,(batch_size, H * W, self.hidden_dim))
        q_h = tf.reshape(q_h,(batch_size, H * W, self.hidden_dim))
        # print(q_h.shape,"q_h")
        k_h = tf.transpose(k_h , perm=[0, 2,1])
        # print(k_h.shape,"ASd")
        a_h = tf.nn.softmax(tf.matmul(q_h, k_h), axis=-1)  # batch_size, H*W, H*W
        # print(a_h.shape)
        v_h = self.layer_v(h)
        v_h = tf.reshape(v_h,(batch_size, H * W, self.hidden_dim))
        z_h = tf.matmul(a_h, v_h)
        # print(z_h.shape,"z_h")

        # memory m attention
        k_m = self.layer_k2(m)
        v_m = self.layer_v2(m)
        k_m = tf.reshape(k_m,(batch_size, H * W, self.hidden_dim))
        v_m = tf.reshape(v_m,(batch_size, H * W, self.hidden_dim))
        k_m = tf.transpose(k_m , perm=[0, 2, 1])
        a_m = tf.nn.softmax(tf.matmul(q_h, k_m), axis=-1)
        # print(a_m.shape, "a_m")

        v_m = self.layer_v2(m)
        v_m = tf.reshape(v_m,(batch_size, H * W, self.hidden_dim))
        z_m = tf.matmul(a_m, v_m) #tf.tranpose(v_m, perm=[0, 2, 1])
        # print(z_m.shape,"z_m")
        z_h = tf.reshape(z_h,(batch_size, H ,W, self.hidden_dim))
        z_m = tf.reshape(z_m,(batch_size, H ,W, self.hidden_dim))

        w_z = tf.concat([z_h, z_m], axis=-1)
        # print(w_z.shape,"w_z")
        Z = self.layer_z(w_z)
        # print(Z.shape,"Z")
        # print(h.shape,"h")
        # Memory Updating
        combined = self.layer_m(tf.concat([Z, h], axis=-1))  # 3 * input_dim
        # print(combined.shape,"com")
        mo, mg, mi = tf.split(combined, 3, axis=-1)
        # print(mo.shape,mg.shape,mi.shape)
        mi = tf.keras.activations.sigmoid(mi)
        new_m = (1 - mi) * m + mi * tf.keras.activations.tanh(mg)
        new_h = tf.keras.activations.sigmoid(mo) * new_m
        # print(new_h.shape,new_m.shape)
    
        return new_h, new_m

