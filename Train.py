import time
import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt
import os

def train(epochs,model,optimizer,train_loader,valid_loader,ckpt_manager,file_name):
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

        if training == True:
            variables = model.trainable_variables
            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss
    df=[]
    ex_val=999999
    early=0
    for epoch in range(epochs+1):
        tt=time.time()
        total_loss, total_val_score = 0, 0
        for inp,targ in train_loader:
            batch_loss = train_step(inp, targ, True)
            total_loss += batch_loss

        total_loss/=len(train_loader)

        for inp,targ in valid_loader:
            batch_loss = train_step(inp, targ, False)
            total_val_score += batch_loss

        total_val_score /=len(valid_loader)
        df.append([epoch,total_loss.numpy(),total_val_score.numpy()])

        # if total_val_score >= ex_val:
        #     early+=1
        # else:
        #     early=0
        # ex_val=total_val_score
        # if early==5:
        #     break


        if epoch%10==0:
            print(f"epochs : {epoch}  total_loss : {total_loss} , total_val_score : {total_val_score} time : {time.time()-tt}" )
        # if epoch%10==0:
        #     print("Save ckpt")
        #     ckpt_manager.save()

    os.makedirs(f'log/{file_name}',exist_ok=True)
    df=pd.DataFrame(df,columns=['epoch','train_loss','val_loss'])
    df.to_csv(f"log/{file_name}/{file_name}.csv",index=False)
    
    
    plt.plot(df.epoch, df.val_loss,'g')
    plt.title("Val_loss")
    plt.savefig(f'log/{file_name}/{file_name}_val.png')
    plt.clf()
    
    plt.plot(df.epoch, df.train_loss,'b')
    plt.title("train_loss")
    plt.savefig(f'log/{file_name}/{file_name}_train.png')
    plt.clf()
    return total_loss.numpy(),total_val_score.numpy()