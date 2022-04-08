import numpy as np
from Metrics import metrics_
import matplotlib.pyplot as plt
import os

def evaluate(batch_size,model,folder_name,file_name):
    path=f"../{folder_name}"
    target_list=[0,8]
    i_time=time.time()
    evaluate_by_image(path,file_name,model,target_list)
    print(f"image_time : {time.time()-i_time}")
    t_time=time.time()
    evaluate_by_time(batch_size,path,file_name)
    print(f"time_time : {time.time() - t_time}")


def evaluate_by_time(batch_size,path,file_name):
    # 2020년 (훈련안시킨거) 모든거 예측하고 매트릭 확인
    win = 7
    total_all = []

    x_test_len = len(os.listdir(f"{path}/test"))

    for k in range(x_test_len):
        before_list = []
        after_list = []
        peak_list = []
        rest_list = []
        # for i in tqdm(list):
        x_test = np.load(f"{path}/2020/{k}.npz")['arr_0']



        for target in range(batch_size - win):
            predict = _predict(model,x_test, target)
            original = x_test[target + 7]

            all = _make_serial(original)
            original = np.array(all)[:, :, :, :, 0]

            # before peak hour - 7~12
            before = compute_metrics(original, predict, 7, 12)
            # peak 12~19
            peak = compute_metrics(original, predict, 12, 19)
            # after 19~21
            after = compute_metrics(original, predict, 19, 21)

            # rest 22~24 , 0~6
            y = original[:, 21:23, :, :]
            y_pred = predict[:, 21:23, :, :]
            # 22~24 0~6 시간대 합치기
            y = np.concatenate((y, original[:, 0:5, :, :]), axis=1)
            y_pred = np.concatenate((y_pred, predict[:, 0:5, :, :]), axis=1)
            # rest 에러 계산
            y = (y) * 100
            y_pred = (y_pred) * 100
            y_flatten = y.flatten()
            y_pred_flatten = y_pred.flatten()

            mape = MAPE(y_flatten, y_pred_flatten, 0)
            rmse = np.sqrt(mean_squared_error(y_flatten, y_pred_flatten))
            mae = mean_absolute_error(y_flatten, y_pred_flatten)

            rest = [rmse, mape, mae]

            # 전체 저장
            before_list.append(before)
            after_list.append(after)
            peak_list.append(peak)
            rest_list.append(rest)

        total_all.append(
            np.array((np.array(before_list), np.array(peak_list), np.array(after_list), np.array(rest_list))))
    total_all = np.array(total_all)
    f = open(f"log/{file_name}/{file_name}_eval_by_time.txt", 'w')
    f.write("rmse mape mae")
    f.write("평균")
    f.write("before")
    f.write(np.mean(total_all[0][0], axis=0))
    f.write("peak")
    f.write(np.mean(total_all[0][1], axis=0))
    f.write("after")
    f.write(np.mean(total_all[0][2], axis=0))
    f.write("rest")
    f.write(np.mean(total_all[0][3], axis=0))
    f.write("표준편차")
    f.write("before")
    f.write(np.std(total_all[0][0], axis=0))
    f.write("peak")
    f.write(np.std(total_all[0][1], axis=0))
    f.write("after")
    f.write(np.std(total_all[0][2], axis=0))
    f.write("rest")
    f.write(np.std(total_all[0][3], axis=0))
    f.close()




def evaluate_by_image(path,file_name,model,target_list):
    x_test = np.load(f"{path}/test/0.npz")['arr_0']
    # target_list [0,8,~]
    #원본데이터
    for target in target_list:

        original=np.array(_make_serial(x_test[target+7]))[:,:,:,:,0]
        predict=_predict(model,x_test,target)
        rmse,mape,mae=(metrics_(original[:,:,:,0],predict[:,:,:,0]))

        plt.clf()
        fig, axes = plt.subplots(2, 7, figsize=(20, 10))
        
        for idx, ax in enumerate(axes[0]):
            ax.imshow(original[idx])
            ax.set_title(f"Original Frame {idx}")
            ax.axis("off")
        for idx, ax in enumerate(axes[1]):
            ax.imshow(predict[idx])
            ax.axis("off")
        # axes[1][0].set_title(f"rmse : {rmse}")
        # axes[1][1].set_title(f"mape : {mape}")
        # axes[1][2].set_title(f"mae : {mae}")
        plt.title(f"rmse : {rmse}, mape : {mape}, mae : {mae}")
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