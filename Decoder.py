from ConvLSTMCEll import ConvLSTMCell
import tensorflow as tf


class Decoder(tf.keras.Model):
    def __init__(self, hidden, dec_num_layers=1, future_len=7):
        super(Decoder, self).__init__()
        self.dec_num_layers = dec_num_layers
        self.future_len = future_len
        self.decoder_input_convlstm = ConvLSTMCell(
            hidden_dim=hidden,
            kernel_size=(3, 3),
            att_hidden_dim=hidden,
            bias=True
        )
        if self.dec_num_layers is not None:
            self.hidden_decoder_layers = [
                ConvLSTMCell(
                    hidden_dim=hidden,
                    kernel_size=(3, 3),
                    att_hidden_dim=hidden,
                    bias=True
                ) for _ in range(dec_num_layers)
            ]
            # self.norm_layers = [
            #     tf.keras.layers.LayerNormalization(axis=-1)
            #     for _ in range(self.dec_num_layers)
            # ]

        self.decoder_output_layer = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=(3, 3),
            padding='same',
            activation='sigmoid'
        )

    def call(self, enc_output):
        if self.dec_num_layers is not None:
            hidden_h_t = []
            hidden_c_t = []
            hidden_m_t = []
            for i in range(self.dec_num_layers):
                hidden_h_t += [self.init_hidden(enc_output[0], i)[0]]
                hidden_c_t += [self.init_hidden(enc_output[0], i)[1]]
                hidden_m_t += [self.init_hidden(enc_output[0], i)[2]]
        outputs = []
        input_tensor = enc_output[0]
        h_t, c_t, m_t = self.init_hidden(input_tensor, 'seq')
        for t in range(self.future_len):
            h_t, c_t, m_t = self.decoder_input_convlstm(
                input_tensor=input_tensor,
                cur_state=[h_t, c_t, m_t]
            )
            input_tensor = h_t
            if self.dec_num_layers is not None:
                for i in range(self.dec_num_layers):
                    hidden_h_t[i], hidden_c_t[i], hidden_m_t[i] = self.hidden_decoder_layers[i](
                        input_tensor=input_tensor,
                        cur_state=[hidden_h_t[i], hidden_c_t[i], hidden_m_t[i]]
                    )
                    # hidden_h_t[i] = self.norm_layers[i](hidden_h_t[i])
                    input_tensor = hidden_h_t[i]
                output = self.decoder_output_layer(hidden_h_t[-1])
            else:
                output = self.decoder_output_layer(h_t)
            outputs += [output]
        outputs = tf.stack(outputs, 1)
        return outputs

    def init_hidden(self, input_tensor, seq):
        if seq == 'seq':
            b, h, w, _ = input_tensor.shape
            h_t, c_t, m_t = self.decoder_input_convlstm.init_hidden(
                batch_size=b,
                image_size=(h, w)
            )
        else:
            b, h, w, _ = input_tensor.shape
            h_t, c_t, m_t = self.hidden_decoder_layers[seq].init_hidden(
                batch_size=b,
                image_size=(h, w)
            )
        return h_t, c_t, m_t