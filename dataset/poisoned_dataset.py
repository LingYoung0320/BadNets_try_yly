# -*- coding: UTF-8 -*- #
"""
@filename:poisoned_dataset.py
@author:Young
@time:2023-12-23
"""
# 负责定义注入攻击的方式

import copy
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

class PoisonedDataset(Dataset):

    def __init__(self, dataset, trigger_label, portion=0.1, mode="train", device=torch.device("cuda"), dataname="mnist"):
        self.class_num = len(dataset.classes)
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx
        self.device = device
        self.dataname = dataname
        self.data, self.targets = self.add_trigger(self.reshape(dataset.data, dataname), dataset.targets, trigger_label, portion, mode)
        self.channels, self.width, self.height = self.__shape_info__()
        # 查看注入的图案
        # self.show_first_poisoned_image()

    def __getitem__(self, item):
        img = self.data[item]
        label_idx = self.targets[item]

        label = np.zeros(10)
        label[label_idx] = 1 # 把num型的label变成10维列表。
        label = torch.Tensor(label)

        img = img.to(self.device)
        label = label.to(self.device)

        return img, label

    def __len__(self):
        return len(self.data)

    def __shape_info__(self):
        return self.data.shape[1:]

    def reshape(self, data, dataname="mnist"):
        if dataname == "mnist":
            new_data = data.reshape(len(data),1,28,28)
        elif dataname == "cifar10":
            new_data = data.reshape(len(data),3,32,32)
        return np.array(new_data)

    def norm(self, data):
        offset = np.mean(data, 0)
        scale  = np.std(data, 0).clip(min=1)
        return (data - offset) / scale

    # 污染数据集
    # targets - 存储了每个样本的原始标签
    # trigger_label - 要更改的新标签
    def add_trigger(self, data, targets, trigger_label, portion, mode):
        print("## generate " + mode + " Bad Imgs")
        new_data = copy.deepcopy(data)
        new_targets = copy.deepcopy(targets)
        perm = np.random.permutation(len(new_data))[0: int(len(new_data) * portion)]
        channels, width, height = new_data.shape[1:]
        for idx in perm: # if image in perm list, add trigger into img and change the label to trigger_label
            # new_targets[idx] = trigger_label
            # i -> i+1
            if new_targets[idx] == 9:
                new_targets[idx]=0
            else:
                new_targets[idx]+=1

            for c in range(channels):
                # 图案1
                # new_data[idx, c, width-4, height-2] = 255
                # new_data[idx, c, width-2, height-2] = 255
                # new_data[idx, c, width-3, height-3] = 255
                # new_data[idx, c, width-2, height-4] = 255
                #图案2
                # new_data[idx, c, width - 2, height - 2] = 255
                # new_data[idx, c, width - 3, height - 2] = 255
                # new_data[idx, c, width - 2, height - 3] = 255
                # new_data[idx, c, width - 3, height - 3] = 255
                #图案3
                new_data[idx, c, width - 2, height - 2] = 255
                new_data[idx, c, width - 3, height - 2] = 255
                new_data[idx, c, width - 2, height - 3] = 255
                new_data[idx, c, width - 3, height - 3] = 255
                new_data[idx, c, width - 4, height - 4] = 255
                new_data[idx, c, width - 4, height - 3] = 255
                new_data[idx, c, width - 4, height - 2] = 255
                new_data[idx, c, width - 2, height - 4] = 255
                new_data[idx, c, width - 3, height - 4] = 255

        print("Injecting Over: %d Bad Imgs, %d Clean Imgs (%.2f)" % (len(perm), len(new_data)-len(perm), portion))
        return torch.Tensor(new_data), new_targets

    def show_first_poisoned_image(self):
        # Show the first poisoned image
        plt.imshow(self.data[0].permute(1, 2, 0).numpy())
        plt.title('First Poisoned Image')
        plt.show()