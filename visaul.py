# -*- codeing = utf-8 -*-
# @Time : 2024/1/21 13:35
# @Author : 李昌杏
# @File : visaul.py
# @Software : PyCharm
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from network import Encoder
from datasets import TeDataset
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
bi=0.3

class Option:
    def __init__(self):
        parser = argparse.ArgumentParser(description="args for model")
        # dataset
        parser.add_argument('--component_path', type=str, default=f"datalist/TE_OBI_com.txt")
        parser.add_argument('--character_path', type=str, default=f"datalist/TE_OBI_cha.txt")
        parser.add_argument("--img_size", default=224)
        parser.add_argument("--batch_size", default=128)
        parser.add_argument("--k", default=10)
        parser.add_argument("--dim", default=768)
        parser.add_argument("--num_class", default=20)
        parser.add_argument("--VIT_pre_weight", default='vit.npz')
        self.parser = parser

    def parse(self):
        return self.parser.parse_args()


def mAP1(loader_cha,loader_com, model_com,model_cha,k,mapper):
    instance_same_class={}
    for i in range(len(mapper)):
        if i < len(mapper) - 1:
            mapper[i] = mapper[i][:-1]
        mapper[i] = mapper[i].split(' ')
        instance_same_class[mapper[i][0]]=[]
        for j in mapper[i][1:]:
            instance_same_class[mapper[i][0]].append(int(j))
    gallery_cha = []
    gallery_reprs_cha = []
    gallery_labels_cha = []
    gallery_name = []
    gallery_name_index = []
    model_com.eval()
    model_cha.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(loader_cha)):
            cha, cha_ori,label,name = data
            cha, label = cha.cuda(), label
            cha_fea=model_cha.embedding(cha).cpu()
            gallery_reprs_cha.append(cha_fea)
            gallery_labels_cha.append(label)
            gallery_name.append(name)
            for i in range(len(name)):
                gallery_name_index.append([batch_idx,i])
            gallery_cha.append(cha_ori)

        gallery_name_index=np.array(gallery_name_index)

        gallery_reprs_cha = F.normalize(torch.cat(gallery_reprs_cha))
        gallery_labels_cha = torch.cat(gallery_labels_cha)
        gallery_cha=torch.cat(gallery_cha)
        gallery_name_index=torch.tensor(gallery_name_index)

        for idx1,(com,com_ori,label,name) in enumerate(tqdm(loader_com)):
            com= com.cuda()
            com_fea=F.normalize(model_com.embedding(bu).cpu())
            ranks = torch.argsort(torch.matmul(com_fea, gallery_reprs_cha.T), dim=1, descending=True).cpu()
            retrievals_cha = gallery_cha[ranks[:, :k]]
            retrievals_name_index = gallery_name_index[ranks[:,:k]]

            for idx, retrieval_cha in enumerate(retrievals_cha):
                s=com_ori[idx]
                l=label[idx]
                p=retrieval_cha
                pic=[]
                s=cv2.copyMakeBorder(np.array(s), 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                s= cv2.copyMakeBorder(s, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                pic.append(cv2.resize(s,(224,224)))
                for _ in range(len(p)):
                    index=retrievals_name_index[idx][_]
                    cha_name=gallery_name[index[0]][index[1]].split('.')[0].split('_')[1]
                    if int(l) in instance_same_class[cha_name]:
                        img_tmp=cv2.copyMakeBorder(np.array(p[_]), 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[0, 255, 0])
                    else:
                        img_tmp=cv2.copyMakeBorder(np.array(p[_]), 3, 3, 3,3, cv2.BORDER_CONSTANT, value=[0, 0, 255])
                    img_tmp = cv2.copyMakeBorder(img_tmp, 1, 1,1, 1, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                    pic.append(cv2.resize(img_tmp,(224,224)))
                img=np.concatenate(pic,axis=1)
                img=np.array(img)
                cv2.imwrite(f'result/{name[idx].replace("/","_")}',img)


def evaluate(args):
    datasets_com= TeDataset(args.data_path1,args.data_path2,m='com',d=True)
    datasets_cha= TeDataset(args.data_path1,args.data_path2,m='cha',d=True)
    model_com = Encoder(args.num_class,checkpoint_path='vit.npz').cuda()
    model_cha = Encoder(args.num_class,checkpoint_path='vit.npz').cuda()

    model_com.load_state_dict(torch.load(f'weights/com_weight.pth'))
    model_cha.load_state_dict(torch.load(f'weights/cha_weight.pth'))

    test_loader_com = DataLoader(datasets_com, batch_size=128*2, shuffle=True,pin_memory=True)
    test_loader_cha = DataLoader(datasets_cha, batch_size=128*2, shuffle=True,pin_memory=True)

    with open('mapper.txt', 'r',encoding='GBK') as f:
        mapper = f.readlines()
        f.close()
    print(mAP1(test_loader_cha,test_loader_com,model_com,model_cha,args.k,mapper ))

if __name__ == '__main__':
    args = Option().parse()
    evaluate(args)
