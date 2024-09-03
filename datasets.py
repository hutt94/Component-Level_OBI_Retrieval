# -*- codeing = utf-8 -*-
# @Time : 2024/1/20 16:43
# @Author : 李昌杏
# @File : datasets.py
# @Software : PyCharm
import random

from torch.utils.data import Dataset
import numpy as np
import cv2
from torchvision.transforms import transforms


def remove_white_space_image(img_np: np.ndarray, padding: int):
    """
    获取白底图片中, 物体的bbox; 此处白底必须是纯白色.
    其中, 白底有两种表示方法, 分别是 1.0 以及 255; 在开始时进行检查并且匹配
    对最大值为255的图片进行操作.
    三通道的图无法直接使用255进行操作, 为了减小计算, 直接将三通道相加, 值为255*3的pix 认为是白底.
    :param img_np:
    :return:
    """
    h, w, c = img_np.shape
    img_np_single = np.sum(img_np, axis=2)
    Y, X = np.where(img_np_single <= 300)  # max = 300
    ymin, ymax, xmin, xmax = np.min(Y), np.max(Y), np.min(X), np.max(X)
    img_cropped = img_np[max(0, ymin - padding):min(h, ymax + padding), max(0, xmin - padding):min(w, xmax + padding),
                  :]
    return img_cropped

def preprocess(image_path,split,ori=False):
        immean = [0.5, 0.5, 0.5]
        imstd = [0.5, 0.5, 0.5]

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(immean, imstd),
        ])

        if split=='com':
            img=cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            img=(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
            img = cv2.resize(img,(224,224))
            img[img>0]=255
        else:
            img=cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            img=(img[:,:,-1])
            img[img>0]=255
            img = cv2.resize(img,(224,224))

        _img = np.stack((img,) * 3, axis=-1)
        _img = remove_white_space_image(_img, 10)
        if ori:
            img=cv2.bitwise_not(img)
            return transform(_img),np.stack((img,) * 3, axis=-1)

        return transform(_img)

class TrDataset(Dataset):
    def __init__(self,path_com,path_cha,num_classes=20):
        with open(path_cha,'r',encoding='GBK') as f:
            path_cha_list=f.readlines()
            f.close()
        with open(path_com,'r',encoding='GBK') as f:
            path_com_list=f.readlines()
            f.close()

        for i in range(len(path_com_list)-1):
            path_com_list[i]=path_com_list[i][:-1]
        for i in range(len(path_cha_list)-1):
            path_cha_list[i]=path_cha_list[i][:-1]

        path_cha_list=path_cha_list+path_com_list

        self.com_class_index={index:[]for index in range(num_classes)}
        self.cha_class_index={index:[]for index in range(num_classes)}
        self.cha_path=[]
        self.com_path=[]
        self.label = []
        self.dict_key=list(self.com_class_index.keys())

        for idx in range(len(path_com_list)):
            blank=path_com_list[idx].split(' ')
            label = blank[-1]
            path = blank[0]
            for _ in blank[1:-1]:
                path += " " + _
            label=int(label)
            self.com_class_index[label].append(idx)
            self.com_path.append(path)
            self.label.append(label)

        for idx in range(len(path_cha_list)):
            blank = path_cha_list[idx].split(' ')
            label = blank[-1]
            path = blank[0]
            for _ in blank[1:-1]:
                path += " " + _
            label = int(label)
            self.cha_class_index[label].append(idx)
            self.cha_path.append(path)

    def __len__(self):return len(self.com_path)

    def __getitem__(self, idx):
        label = self.label[idx]
        com_path = self.com_path[idx]

        neg_label = random.choice(self.dict_key)
        while neg_label==label:
            neg_label = random.choice(self.dict_key)

        com_pos_path=self.com_path[random.choice(self.com_class_index[label])]
        cha_path=self.cha_path[random.choice(self.cha_class_index[label])]
        cha_neg_path=self.cha_path[random.choice(self.cha_class_index[neg_label])]

        com=preprocess(com_path,'com')
        com_pos=preprocess(com_pos_path,'com')
        # print(cha_path,cha_neg_path,com_path,com_pos_path)
        if 'a' in cha_path:
            cha=preprocess(cha_path[:-6]+'.png','cha')
        else:
            cha=preprocess(cha_path,'cha')

        if 'a' in cha_neg_path:
            cha_neg=preprocess(cha_neg_path[:-6]+'.png','cha')
        else:
            cha_neg=preprocess(cha_neg_path,'cha')

        return com,com_pos,cha,cha_neg,label

class TeDataset(Dataset):
    def __init__(self, path_com, path_cha, num_classes=20,m='com',d=False):
        self.d = d
        with open(path_cha, 'r', encoding='GBK') as f:
            path_cha_list = f.readlines()
            f.close()
        with open(path_com, 'r', encoding='GBK') as f:
            path_com_list = f.readlines()
            f.close()

        for i in range(len(path_com_list) - 1):
            path_com_list[i] = path_com_list[i][:-1]
        for i in range(len(path_cha_list) - 1):
            path_cha_list[i] = path_cha_list[i][:-1]

        path_cha_list = path_cha_list + path_com_list

        self.com_class_index = {index: [] for index in range(num_classes)}
        self.cha_class_index = {index: [] for index in range(num_classes)}
        self.cha_path = []
        self.com_path = []
        self.com_label = []
        self.cha_label = []
        self.dict_key = list(self.com_class_index.keys())

        for idx in range(len(path_com_list)):
            blank = path_com_list[idx].split(' ')
            label = blank[-1]
            path = blank[0]
            for _ in blank[1:-1]:
                path += " " + _
            label = int(label)
            self.com_class_index[label].append(idx)
            self.com_path.append(path)
            self.com_label.append(label)

        for idx in range(len(path_cha_list)):
            blank = path_cha_list[idx].split(' ')
            label = blank[-1]
            path = blank[0]
            for _ in blank[1:-1]:
                path += " " + _
            label = int(label)
            self.cha_class_index[label].append(idx)
            self.cha_path.append(path)
            self.cha_label.append(label)

        self.m = m

    def __len__(self):
        if self.m=='com':
            return len(self.com_path)
        return len(self.cha_path)

    def __getitem__(self, idx):
        if self.m=='cha':
            label=self.cha_label[idx]
            path=self.cha_path[idx]
            if 'a' in path:
                cha,ori = preprocess(path[:-6] + '.png', 'cha',True)
            else:
                cha,ori = preprocess(path, 'cha',True)
            if self.d:
                name=path.split('/')
                name=name[-1].split(".")[0]
                if 'a' in name:
                    name = name.replace("_a","")
                return cha,ori,label,name
            return cha,label
        else:
            label=self.com_label[idx]
            path=self.com_path[idx]
            com,ori = preprocess(path, 'com',True)
            if self.d:
                name=path.split('/')
                name=name[-1].split(".")[0]
                if 'a' in name:
                    name = name.replace("_a","")
                return com,ori,label,name
            return com,label
