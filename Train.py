import time
import tensorflow as tf
def train(epochs,model,optimizer,train_loader,valid_loader,ckpt_manager):
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

    for epoch in range(epochs+1):
        tt=time.time()
        total_loss, total_val_score = 0, 0
        for inp,targ in train_loader:
            batch_loss = train_step(inp, targ, True)
            total_loss += batch_loss

        total_loss/=len(train_loader)

        for inp,targ in valid_loader:
            batch_loss = train_step(inp, targ, True)
            total_val_score += batch_loss

        total_val_score /=len(valid_loader)

        print(f"epochs : {epoch}  total_loss : {total_loss} , total_val_score : {total_val_score} time : {time.time()-tt}" )
        if epoch%100==0:
            print("Save ckpt")
            ckpt_manager.save()