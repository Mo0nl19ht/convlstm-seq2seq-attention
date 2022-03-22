import tensorflow as tf
from tensorflow import keras
import os
import time
from Dataloader import Dataloader
from Seq2Seq import Seq2Seq
from Train import train
import mlflow


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
#(24,31,1) kang 배치 32
# (24,30,1) olym
# img_shape=(288,17,1)

# mlflow.tensorflow.autolog()


params = {
    'num' : 31,
    'batch_size' : 64,
    'epochs' : 3000,
    'num_layer': 4,
    'lr' : 0.00005,
    'loss_f' : 'mse',
    'filter_size' : 64
}

folder_name="npz_gray_kang"
checkpoint_path = f'./seq2seq/{folder_name}/'
os.makedirs(checkpoint_path,exist_ok=True)
#npz_x의 갯수

val_index=[params['num']-1]
train_index=[]
for i in list(range(params['num'])):
    if i not in val_index:
        train_index.append(i)

train_index=list(set(train_index))
val_index=list(set(val_index))
train_loader = Dataloader(train_index,folder_name)
valid_loader = Dataloader(val_index,folder_name)

# model = Seq2Seq(int(params['filter_size']/4), params['num_layer'], params['num_layer'])

# optimizer = tf.keras.optimizers.Adam(params['lr'])

# ckpt,ckpt_manager = make_checkpoint(checkpoint_path,model,optimizer)

# file_name=f"layernorm_{folder_name}_{params['filter_size']}_{params['lr']}_{params['loss_f']}_{params['num_layer']}_{params['epochs']}"
# print(file_name)

#mlflow 실험 설정   
experiment_name=f"Convl_{folder_name}"
experiment=mlflow.get_experiment_by_name(experiment_name)
# MLFLOW_TRACKING_URI="http://0.0.0.0:8888"
if experiment == None:
    print(f"{experiment_name} 실험을 생성합니다")
    experiment_id = mlflow.create_experiment(experiment_name)
else:
    experiment_id=experiment.experiment_id

lr_li=[0.0005, 0.0001,0.00001,0.00005]
for i in range(len(lr_li)):
    print(i)
    params['lr']=lr_li[i]
    model = Seq2Seq(int(params['filter_size']/4), params['num_layer'], params['num_layer'])
# tf.keras.optimizers.schedules.ExponentialDecay(params['lr'],decay_steps=params['epochs']//10,decay_rate=0.96)
    optimizer = tf.keras.optimizers.Adam(params['lr'])

    ckpt,ckpt_manager = make_checkpoint(checkpoint_path,model,optimizer)

    file_name=f"layernorm_{folder_name}_{params['filter_size']}_{params['lr']}_{params['loss_f']}_{params['num_layer']}_{params['epochs']}"
    print(file_name)

    train_loss,val_loss=train(params['epochs'],model,optimizer,train_loader,valid_loader,ckpt_manager,file_name)

    with mlflow.start_run(experiment_id=experiment_id) as run:
        # mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        # print(mlflow.get_tracking_uri(),"트래킹 uri")

        mlflow.log_params(params)
        mlflow.log_artifacts(f'log/{file_name}')
        # mlflow.set_tag("release.version", "2.2.0")
        
        mlflow.log_metrics({"train_loss": train_loss,"val_loss":val_loss})

        # tf.saved_model.save(model, "./models")

        # mlflow.tensorflow.log_model(model,file_name)
        # mlflow.tensorflow.save_model(model)
        # mlflow.log_metrics(metrics)

    model.save_weights(f'result/model/{file_name}.h5')
