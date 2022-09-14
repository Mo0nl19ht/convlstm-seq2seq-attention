import Evaluate

def  make_eval_image(model,x_test,target_list):
    originals=[]
    predicts=[]
    # target_list=[0,8,16]


    #원본데이터
    original=x_test[target+7]

    all=[]
    for img in original:
        one_img=[]
        for pixels in img:
            pixel=[]
            for gray in pixels:
                pixel.append([gray,gray,gray])
            one_img.append(pixel)
        all.append(one_img)
    original=np.array(all)[:,:,:,:,0]

    predicts=Evaluate.make_predict(model, x_test ,target,original)

    fig, axes = plt.subplots(model_num+1, 7, figsize=(20, 10))
    # Plot the original frames.
    for idx, ax in enumerate(axes[0]):
        #inverse여서 1에서 빼준다
        ax.imshow((original[idx]))
        ax.set_title(f"Original Frame {idx}")
        ax.axis("off")

    for i in range(model_num):
        for idx, ax in enumerate(axes[i+1]):
            ax.imshow(predicts[i][idx])
            ax.set_title(f"Predicted Frame {idx}")
            ax.axis("off")