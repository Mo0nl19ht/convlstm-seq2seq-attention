import numpy as np
from Metrics import metrics_

def evaluate(model,file_name):
    path=f"../{file_name}"
    x_test=np.load(f"{path}/test/0.npz")['arr_0']
    target_list=[0,8]
    evaluate_by_image(file_name,model,x_test,target_list)


def evaluate_by_image(file_name,model,x_test,target_list):
    # target_list [0,8,~]
    #원본데이터
    for target in target_list:

        original=np.array(_make_serial(x_test[target+7]))[:,:,:,:,0]
        predict=_predict(model,x_test,target)
        rmse,mape,mae=(metrics_(original[:,:,:,0],predict[:,:,:,0]))

        plt.clf()
        fig, axes = plt.subplots(2, 7, figsize=(20, 10))
        plt.title(f"rmse : {rmse}, mape : {mape}, mae : {mae}")
        for idx, ax in enumerate(axes[0]):
            ax.imshow(original[idx])
            ax.set_title(f"Original Frame {idx}")
            ax.axis("off")
        for idx, ax in enumerate(axes[1]):
            ax.imshow(predict[idx])
            ax.set_title(f"Predicted Frame {idx}")
            ax.axis("off")
        plt.savefig(f'log/{file_name}/{file_name}_eval_image_target_{target}.png')
        plt.clf()
        


def _predict(model, x_test ,target):
    for idx in range(7):
        a=np.expand_dims(x_test[target+idx], axis=0)
        #1 7 24 31 1
        prd=model(a)

        #gray에서 이미지보여주려고 ch3만듬
        all=[]
        #예측된거 마지막꺼만 가져옴
        for img in prd[0][-1]:
            pixel=[]
            for gray in img:
                pixel.append([gray,gray,gray])
            all.append(pixel)
        all=np.array(all)[:,:,:,0]
    
        if idx==0:
            predict=np.expand_dims(all,axis=0)
        else:
            predict = np.concatenate((predict, np.expand_dims(all,axis=0)), axis=0)
    return predict


def _make_serial(original):
    all=[]
    for img in original:
        one_img=[]
        for pixels in img:
            pixel=[]
            for gray in pixels:
                pixel.append([gray,gray,gray])
            one_img.append(pixel)
        all.append(one_img)
    return all