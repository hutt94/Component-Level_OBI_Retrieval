
# -*- codeing = utf-8 -*-
# @Time : 2024/1/22 9:49
# @Author : 李昌杏
# @File : valid.py
# @Software : PyCharm
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from torch import nn
import utils
from network import Encoder
from datasets import TeDataset
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Option:
    def __init__(self):
        parser = argparse.ArgumentParser(description="args for model")
        # dataset
        parser.add_argument('--componet_path', type=str, default=f"datalist/TE_OBI_com.txt")
        parser.add_argument('--character_path', type=str, default=f"datalist/R_OBI_cha.txt")
        parser.add_argument("--img_size", default=224)
        parser.add_argument("--batch_size", default=128)
        parser.add_argument("--k", default=50)
        parser.add_argument("--dim", default=768)
        parser.add_argument("--num_class", default=20)
        parser.add_argument("--VIT_pre_weight", default='vit.npz')
        self.parser = parser

    def parse(self):
        return self.parser.parse_args()

def eval_AP_inner(inst_id, scores, gt_labels,gallery_name,instance_same_class, top=None):
    pos_flag=[]
    for index in gt_labels:
        cha_name = gallery_name[index[0]][index[1]].split('.')[0].split('_')[1]
        if int(inst_id) in instance_same_class[cha_name]:
            pos_flag.append(True)
        else:pos_flag.append(False)

    pos_flag=np.array(pos_flag)
    tot = scores.shape[0]  # total retrieved samples
    tot_pos = np.sum(pos_flag)  # total true position
    sort_idx = np.argsort(-scores)
    tp = pos_flag[sort_idx]  # sorted true positive
    fp = np.logical_not(tp)  # sorted false positive

    if top is not None:
        top = min(top, tot)
        tp = tp[:top]  # select top-k true position
        fp = fp[:top]
        tot_pos = min(top, tot_pos)

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    try:
        rec = tp / tot_pos
        prec = tp / (tp + fp)
    except:
        print(inst_id, tot_pos)
        return np.nan

    ap = utils.VOCap(rec, prec)
    return ap

def eval_precision(inst_id, scores, gt_labels,gallery_name,instance_same_class, top=100):
    pos_flag = []
    for index in gt_labels:
        cha_name = gallery_name[index[0]][index[1]].split('.')[0].split('_')[1]
        if int(inst_id) in instance_same_class[cha_name]:
            pos_flag.append(True)
        else:
            pos_flag.append(False)

    pos_flag = np.array(pos_flag)
    tot = scores.shape[0]

    top = min(top, tot)

    sort_idx = np.argsort(-scores)
    return np.sum(pos_flag[sort_idx][:top]) / top

def mAP(loader_cha,loader_com, model_com,model_cha,k,mapper):
    instance_same_class={}
    for i in range(len(mapper)):
        if i < len(mapper) - 1:
            mapper[i] = mapper[i][:-1]
        mapper[i] = mapper[i].split(' ')
        instance_same_class[mapper[i][0]]=[]
        for j in mapper[i][1:]:
            instance_same_class[mapper[i][0]].append(int(j))
    gallery_reprs_cha = []
    gallery_reprs_com = []
    gallery_reprs_label = []
    gallery_name = []
    gallery_name_index = []
    model_cha.eval()
    model_com.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(loader_cha)):
            cha, cha_ori,label,name = data
            cha, label = cha.cuda(), label
            cha_fea=model_cha.embedding(cha).cpu()
            gallery_reprs_cha.append(cha_fea)
            gallery_name.append(name)
            for i in range(len(name)):
                gallery_name_index.append([batch_idx,i])

        gallery_name_index=np.array(gallery_name_index)

        gallery_reprs_cha = F.normalize(torch.cat(gallery_reprs_cha))
        gallery_name_index=torch.tensor(gallery_name_index)

        for idx1,(com,com_ori,label,name) in enumerate(tqdm(loader_com)):
            com= com.cuda()
            com_fea=model_com.embedding(com).cpu()
            gallery_reprs_com.append(com_fea)
            gallery_reprs_label.append(label)


        gallery_reprs_com = F.normalize(torch.cat(gallery_reprs_com))
        gallery_reprs_label = torch.cat(gallery_reprs_label)

        test_features_com = nn.functional.normalize(gallery_reprs_com, dim=1, p=2)
        test_features_cha = nn.functional.normalize(gallery_reprs_cha, dim=1, p=2)

        sim = torch.mm(test_features_com, test_features_cha.T)
        k = {'map': 10, 'precision': 10}

        mean_mAP = []
        for fi in range(len(gallery_reprs_label)):
            mapi = eval_AP_inner(gallery_reprs_label[fi], sim[fi], gallery_name_index,gallery_name,instance_same_class,top=k['map'])
            mean_mAP.append(mapi)

        mean_prec = []
        for fi in range(len(gallery_reprs_label)):
            prec = eval_precision(gallery_reprs_label[fi], sim[fi], gallery_name_index,gallery_name,instance_same_class,top=k['precision'])
            mean_prec.append(prec)

        print('map{}: {:.4f} prec{}: {:.4f}'.format(k['map'], np.mean(mean_mAP), k['precision'], np.nanmean(mean_prec)))

        k = {'map': 50, 'precision': 50}

        mean_mAP = []
        for fi in range(len(gallery_reprs_label)):
            mapi = eval_AP_inner(gallery_reprs_label[fi], sim[fi], gallery_name_index,gallery_name,instance_same_class,top=k['map'])
            mean_mAP.append(mapi)

        mean_prec = []
        for fi in range(len(gallery_reprs_label)):
            prec = eval_precision(gallery_reprs_label[fi], sim[fi], gallery_name_index,gallery_name,instance_same_class,top=k['precision'])
            mean_prec.append(prec)

        print('map{}: {:.4f} prec{}: {:.4f}'.format(k['map'], np.mean(mean_mAP), k['precision'], np.nanmean(mean_prec)))

def evaluate(args):
    datasets_com= TeDataset(args.componet_path,args.character_path,m='com',d=True)
    datasets_cha= TeDataset(args.componet_path,args.character_path,m='cha',d=True)
    model_com = Encoder(args.num_class,checkpoint_path='vit.npz').cuda()
    model_cha = Encoder(args.num_class,checkpoint_path='vit.npz').cuda()

    model_com.load_state_dict(torch.load(f'weights/com_weight.pth'))
    model_cha.load_state_dict(torch.load(f'weights/cha_weight.pth'))

    test_loader_com = DataLoader(datasets_com, batch_size=256, shuffle=True,pin_memory=True)
    test_loader_cha = DataLoader(datasets_cha, batch_size=256, shuffle=True,pin_memory=True)

    with open('mapper.txt', 'r') as f:
        mapper = f.readlines()
        f.close()
    print(mAP(test_loader_cha,test_loader_com,model_com,model_cha,args.k,mapper ))

if __name__ == '__main__':
    args = Option().parse()
    evaluate(args)
