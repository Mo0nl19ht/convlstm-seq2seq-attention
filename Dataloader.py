
import math

import numpy as np
from tensorflow.keras.utils import Sequence



class Dataloader(Sequence):

    def __init__(self, data_list,folder_name):
        self.data_list = data_list
        self.folder_name = folder_name

    def __len__(self):
        return math.ceil(len(self.data_list))

    # batch 단위로 직접 묶어줘야 함
    def __getitem__(self, idx):
        # sampler의 역할(index를 batch_size만큼 sampling해줌)
        x_path = f"../{self.folder_name}/batch/x/"
        y_path = f"../{self.folder_name}/batch/y/"
        return np.load(f"{x_path}{self.data_list[idx]}.npz")['x'], np.load(f"{y_path}{self.data_list[idx]}.npz")['y']