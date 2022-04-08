import numpy as np
from Metrics import metrics_ , compute_metrics, MAPE
import matplotlib.pyplot as plt
import time
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error,mean_absolute_error
import os

def evaluate(batch_size,model,folder_name,file_name):
    path=f"../{folder_name}"
    target_list=[0,8]
    i_time=time.time()
    evaluate_by_image(path,file_name,model,target_list)
    print(f"image_time : {time.time()-i_time}")
    s_time=time.time()
    evaluate_by_sector(path, file_name, model, batch_size)
    print(f"sector_time : {time.time() - s_time}")
    t_time=time.time()
    rmse,mape,mae=evaluate_all_test(path, file_name, model, batch_size)
    print(f"time_time : {time.time() - t_time}")
    return rmse,mape,mae

def evaluate_by_sector(path, file_name, model, batch_size):
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
        x_test = np.load(f"{path}/test/{k}.npz")['arr_0']



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
    f = open(f"log/{file_name}/{file_name}_eval_by_sector.txt", 'w')
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
        
def evaluate_all_test(path, file_name, model, batch_size):
    win=7
    total_7 = []
    x_test_len = len(os.listdir(f"{path}/test"))
    for k in range(x_test_len):
        times = []
        x_test = np.load(f"{path}/test/{k}.npz")['arr_0']

        for target in range(0, batch_size - win, win):
            predict = _predict(model, x_test, target)
            original = x_test[target + 7]
            all = _make_serial(original)
            original = np.array(all)[:, :, :, :, 0]
            time = []
            for i in range(1, 25):
                time.append(compute_metrics(original, predict, i, i + 1, is_pval=0))

            # 전체 저장
            times.append(np.array(time))

        total_7.append(np.array(times))
    total_7 = np.array(total_7)
    total_7 = total_7.reshape(-1, 24, 4)

    rmse_list = []
    mape_list = []
    mae_list = []
    # pval_list = []

    for time in range(24):
        # rmse
        rmse_list.append(np.mean(np.sqrt(total_7[:, time, 0].astype(float))))
        # mape
        mape_list.append(np.mean(total_7[:, time, 1]))
        # mae
        mae_list.append(np.mean(total_7[:,time,2]))
        # p_value
        # pval_list.append(np.mean(total_7[:, time, 3]))

    rmse_std = []
    mape_std = []
    mae_std = []

    for time in range(24):
        # rmse
        rmse_std.append(np.std(np.sqrt(total_7[:, time, 0].astype(float)), axis=0))
        # mape
        mape_std.append(np.std(total_7[:, time, 1], axis=0))
        # mae
        mae_std.append(np.std(total_7[:, time, 2], axis=0))

    make_artifact(file_name,rmse_list,"rmse")
    make_artifact(file_name,mape_list,"mape")
    make_artifact(file_name,mae_list,"mae")
    # make_artifact(file_name,pval_list,"p_val")
    make_artifact(file_name,rmse_std,"rmse_std")
    make_artifact(file_name,mape_std,"mape_std")
    make_artifact(file_name,mae_std,"mae_std")

    return np.mean(rmse_list),np.mean(mape_list),np.mean(mae_list)





def make_artifact(file_name,metric_list,metric):
    plt.clf()
    plt.title(f"{metric}")
    plt.plot(range(24), metric_list)
    plt.savefig(f'log/{file_name}/{file_name}_eval_time_{metric}.png')
    plt.clf()

    f = open(f"log/{file_name}/{file_name}_eval_time_{metric}.txt", 'w')
    f.write(f"time,{metric}")
    for i,data in enumerate(metric_list):
        f.write(f"{i},{data}")
    f.close()

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