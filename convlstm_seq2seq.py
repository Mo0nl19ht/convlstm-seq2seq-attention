import tensorflow as tf
from tensorflow import keras
import os
import time
from Dataloader import Dataloader
from Seq2Seq import Seq2Seq
from Train import train

def MAPE(y_test, y_pred):
    # print(y_test.shape, y_pred.shape)
    y_t = tf.where(tf.math.equal(y_test, 0), 1e-17, y_test)
    y_p = tf.where(tf.math.equal(y_test, 0), 1e-17, y_pred)
    return keras.losses.MAPE(y_t, y_p)

def make_checkpoint(checkpoint_path,model,optimizer):
    os.makedirs(checkpoint_path, exist_ok=True)
    ckpt = tf.train.Checkpoint(
        Seq2Seq=model,
        optimizer=optimizer
    )
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=100)
    # start_epoch = 0
    # if ckpt_manager.latest_checkpoint:
    #     start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    #     ckpt.restore(ckpt_manager.latest_checkpoint)
    #     print(ckpt_manager.latest_checkpoint)
    return ckpt,ckpt_manager

#(24,14,1) fix
#(24,31,1) kang
# (24,30,1) olym

folder_name="npz_gray_7_64_fix"
checkpoint_path = './seq2seq/kang/'
#npz_x의 갯수
num=16
batch_size=64
img_shape=(24,14,1)
epochs = 12000
num_layer=4
lr = 0.00001
loss_f="mse"
drop=0.1

val_index=[3]
train_index=[]
for i in list(range(num)):
    if i not in val_index:
        train_index.append(i)

train_index=list(set(train_index))
val_index=list(set(val_index))
train_loader = Dataloader(train_index,folder_name)
valid_loader = Dataloader(val_index,folder_name)

filter_size=64

model = Seq2Seq(int(filter_size/4), num_layer, num_layer)
optimizer = tf.keras.optimizers.Adam(lr)

ckpt,ckpt_manager = make_checkpoint(checkpoint_path,model,optimizer)

file_name=f'seq2seq_디코더까지layernorm_inside_{filter_size}_{lr}_{loss_f}_{num_layer}_{epochs}_{drop}'
print(file_name)
train(epochs,model,optimizer,train_loader,valid_loader,ckpt_manager,file_name)
model.save_weights(f'{file_name}.h5')
