from google.protobuf.descriptor import TypeTransformationError
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, metrics
import io
from PIL import Image
import math
from tensorflow.keras.utils import Sequence
import os
from tensorflow.python.client import device_lib
import random

##로드하는법
# new_model = keras.models.load_model('path_to_my_model')

folder_name="npz_gray_kang"
#npz_x의 갯수
num=31
batch_size=32
#(24,14,1) fix
#(24,31,1) kang
# (24,30,1) olym
img_shape=(24,31,1)


class Dataloader(Sequence):

    def __init__(self, data_list):
        self.data_list=data_list
        # self.batch_size = batch_size
    
    def __len__(self):
        return math.ceil(len(self.data_list))

		# batch 단위로 직접 묶어줘야 함
    def __getitem__(self, idx):
				# sampler의 역할(index를 batch_size만큼 sampling해줌)
        x_path=f"../{folder_name}/batch/x/"
        y_path=f"../{folder_name}/batch/y/"
        return np.load(f"{x_path}{self.data_list[idx]}.npz")['x'] , np.load(f"{y_path}{self.data_list[idx]}.npz")['y']


class ConvLSTMCell(tf.keras.Model):
    def __init__(self, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.hidden_dim = (hidden_dim)
        
        self.kernel_size = kernel_size
        self.bias = bias
        
        self.conv = tf.keras.layers.Conv2D(
            filters = 4 * self.hidden_dim,
            kernel_size = self.kernel_size,
            padding = 'same',
            use_bias = self.bias
        )
        
    # convlstm계산 과정
    # 먼저 해당시점의 인코더 h_가중치 이어붙이고 conv연산 
    # 시그모이드연산을 함 - 이게 lstm에 해당되는건가..?
    def call(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = tf.concat([input_tensor, h_cur], axis=3)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = tf.split(combined_conv, num_or_size_splits=4, axis=-1)
        i = tf.keras.activations.sigmoid(cc_i)
        f = tf.keras.activations.sigmoid(cc_f)
        o = tf.keras.activations.sigmoid(cc_o)
        g = tf.keras.activations.tanh(cc_g)
        
        c_next = f*c_cur+i*g
        h_next = o*tf.keras.activations.tanh(c_next)
        
        return h_next, c_next
        
    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        # print("@@@")
        # print(batch_size, height, width, self.hidden_dim)
        # print("@@@")
        return (tf.zeros([batch_size, height, width, self.hidden_dim]),
                tf.zeros([batch_size, height, width, self.hidden_dim]))

class Encoder(tf.keras.Model):
    def __init__(self, hidden, enc_num_layers=1):
        super(Encoder, self).__init__()
        self.enc_num_layers = enc_num_layers
        self.encoder_input_convlstm = ConvLSTMCell(
            hidden_dim=hidden,
            kernel_size=(3, 3),
            bias=True
        )
        if self.enc_num_layers is not None:
            self.hidden_encoder_layers = [
                ConvLSTMCell(
                    hidden_dim=hidden,
                    kernel_size=(3, 3),
                    bias=True
                ) for _ in range(self.enc_num_layers)
            ]
        
    def call(self, enc_input):
        # 모든 레이어 초기화 하기
        h_t, c_t = self.init_hidden(enc_input, 'seq')
        if self.enc_num_layers is not None:
            hidden_h_t = []
            hidden_c_t = []
            for i in range(self.enc_num_layers):
                hidden_h_t += [self.init_hidden(h_t, i)[0]]
                hidden_c_t += [self.init_hidden(h_t, i)[1]]
        
        seq_len = enc_input.shape[1]
        for t in range(seq_len):
            h_t, c_t = self.encoder_input_convlstm(
                #타임 시퀀스 하나를 넣는다
                
                input_tensor=enc_input[:, t, :, :, :],
                # cur_state는 현재 가중치
                cur_state=[h_t, c_t]
            )
            # input conv셀에서 나온 h_t를 인풋으로 각 인코더 convlstm레이어에 넣어서 연산한다
            input_tensor = h_t
            if self.enc_num_layers is not None:
                for i in range(self.enc_num_layers):
                    hidden_h_t[i], hidden_c_t[i] = self.hidden_encoder_layers[i](
                        input_tensor=input_tensor,
                        cur_state=[hidden_h_t[i], hidden_c_t[i]]
                    )
                    input_tensor = hidden_h_t[i]
        
        if self.enc_num_layers is not None:
            return hidden_h_t[-1], hidden_c_t[-1]
        else:
            return h_t, c_t
    
    def init_hidden(self, input_tensor, seq):
        # 이거는 인코더 맨 처음 초기화할때만 사용
        if seq == 'seq':
            #32,7,28,31,1
            b, seq_len, h, w, _ = input_tensor.shape
            #[batch_size, height, width, self.hidden_dim] 처음엔 h_t, c_t똑같음
            h_t, c_t = self.encoder_input_convlstm.init_hidden(
                batch_size=b,
                image_size=(h, w)
            )
        else:
            # h_t([batch_size, height, width, self.hidden_dim])가 들어옴 input으로
            b, h, w, _ = input_tensor.shape
            # seq는 인코더 레이어 인덱스
            # 각 convlstm 셀 초기화 시키고 그 값 반환
            h_t, c_t = self.hidden_encoder_layers[seq].init_hidden(
                batch_size=b,
                image_size=(h, w)
            )
        return h_t, c_t

class Decoder(tf.keras.Model):
    def __init__(self, hidden, dec_num_layers=1, future_len=7):
        super(Decoder, self).__init__()
        self.dec_num_layers = dec_num_layers
        self.future_len = future_len
        self.decoder_input_convlstm = ConvLSTMCell(
            hidden_dim=hidden,
            kernel_size=(3, 3),
            bias=True
        )
        if self.dec_num_layers is not None:
            self.hidden_decoder_layers = [
                ConvLSTMCell(
                    hidden_dim=hidden,
                    kernel_size=(3, 3),
                    bias=True
                ) for _ in range(dec_num_layers)
            ]
            
        self.decoder_output_layer = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=(3,3),
            padding='same',
            activation='sigmoid'
        )
        
    def call(self, enc_output):
        if self.dec_num_layers is not None:
            hidden_h_t = []
            hidden_c_t = []
            for i in range(self.dec_num_layers):
                hidden_h_t += [self.init_hidden(enc_output[0], i)[0]]
                hidden_c_t += [self.init_hidden(enc_output[0], i)[1]]
                
        outputs = []
        input_tensor = enc_output[0]
        h_t, c_t = self.init_hidden(input_tensor, 'seq')
        for t in range(self.future_len):
            h_t, c_t=self.decoder_input_convlstm(
                input_tensor=input_tensor,
                cur_state=[h_t, c_t]
            )
            input_tensor = h_t
            if self.dec_num_layers is not None:
                for i in range(self.dec_num_layers):
                    hidden_h_t[i], hidden_c_t[i] = self.hidden_decoder_layers[i](
                        input_tensor=input_tensor,
                        cur_state=[hidden_h_t[i], hidden_c_t[i]]
                    )
                    input_tensor=hidden_h_t[i]
                output = self.decoder_output_layer(hidden_h_t[-1])
            else:
                output = self.decoder_output_layer(h_t)
            outputs += [output]
        outputs = tf.stack(outputs, 1)
        
        return outputs
    
    def init_hidden(self, input_tensor, seq):
        if seq == 'seq':
            b, h, w, _ = input_tensor.shape
            h_t, c_t = self.decoder_input_convlstm.init_hidden(
                batch_size=b,
                image_size=(h, w)
            )
        else:
            b, h, w, _ = input_tensor.shape
            h_t, c_t = self.hidden_decoder_layers[seq].init_hidden(
                batch_size=b,
                image_size=(h, w)
            )
        return h_t, c_t

class Seq2Seq(tf.keras.Model):
    def __init__(self, hidden, enc_num_layers=1, dec_num_layers=1):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(hidden, enc_num_layers)
        self.decoder = Decoder(hidden, dec_num_layers)
        
    def call(self, enc_input):
        enc_output = self.encoder(enc_input)
        dec_output = self.decoder(enc_output)
        
        return dec_output

def MAPE(y_test, y_pred):
    # print(y_test.shape, y_pred.shape)
    y_t=tf.where(tf.math.equal(y_test, 0),1e-17,y_test)
    y_p=tf.where(tf.math.equal(y_test, 0),1e-17,y_pred)
    

    return keras.losses.MAPE(y_t,y_p)


val_index=[3]
train_index=[]
for i in list(range(num)):
    if i not in val_index:
        train_index.append(i)

train_index=list(set(train_index))
val_index=list(set(val_index))


train_loader = Dataloader(train_index)
valid_loader = Dataloader(val_index)

# inp = layers.Input(shape=(None,img_shape[0],img_shape[1],img_shape[2]))
num_layer=1
model = Seq2Seq(16, num_layer, num_layer)
filter_size=64
lr = 0.005
loss_f="mse"
optimizer = tf.keras.optimizers.Adam(lr)


@tf.function
def loss_function(output, target):
    mse = tf.math.reduce_mean(tf.keras.losses.MSE(output, target))
    return mse


@tf.function
def train_step(inp, targ, training):
    loss = 0
    with tf.GradientTape() as tape:
        output = model(inp)
        for t in range(targ.shape[1]):
            loss += loss_function(targ[:, t], output[:, t])
            
    batch_loss = (loss / int(targ.shape[1]))
    
    if training==True:
        variables = model.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        
    return batch_loss

checkpoint_path = './seq2seq/kang/'
os.makedirs(checkpoint_path, exist_ok=True)
ckpt = tf.train.Checkpoint(
    Seq2Seq=model, 
    optimizer=optimizer
)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=100)

start_epoch = 0
# if ckpt_manager.latest_checkpoint:
#     start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
#     ckpt.restore(ckpt_manager.latest_checkpoint)
#     print(ckpt_manager.latest_checkpoint)

EPOCHS = 1000
import time

for epoch in range(EPOCHS+1):
    tt=time.time()
    total_loss, total_val_score = 0, 0
    # tqdm_dataset = (enumerate(train_dataset))
    for inp,targ in train_loader:
        batch_loss = train_step(inp, targ, True)
        total_loss += batch_loss
        
    total_loss/=len(train_loader)

    for inp,targ in valid_loader:
        batch_loss = train_step(inp, targ, True)
        total_val_score += batch_loss

    total_val_score /=len(valid_loader)

    print(f"epochs : {epoch}  total_loss : {total_loss} , total_val_score : {total_val_score} time : {time.time()-tt}" )
    if epoch%10==0:
        print("Save ckpt")
        ckpt_manager.save()


model.save_weights(f'seq2seq_KangByeon_{filter_size}_lay5_{lr}_{loss_f}_{num_layer}_{epoch}.h5')
