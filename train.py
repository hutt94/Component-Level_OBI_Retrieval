# -*- codeing = utf-8 -*-
# @Time : 2024/1/20 18:53
# @Author : 李昌杏
# @File : train.py
# @Software : PyCharm
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn
import losses
import utils
from network import Encoder
from datasets import TrDataset
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
bi = 0.3


class Option:
    def __init__(self):
        parser = argparse.ArgumentParser(description="args for model")
        # dataset
        parser.add_argument('--component_path', type=str, default=f"datalist/Tr_OBI_com.txt")
        parser.add_argument('--character_path', type=str, default=f"datalist/Tr_OBI_cha.txt")
        parser.add_argument("--seed", default=1234)
        # train
        parser.add_argument("--img_size", default=224)
        parser.add_argument("--epoch", default=100)
        parser.add_argument("--warmup_epochs", default=3)
        parser.add_argument("--batch_size", default=32)
        parser.add_argument("--lr", default=3e-3)
        parser.add_argument("--min_lr", default=1e-4)
        parser.add_argument("--weight_decay", default=0.04)
        parser.add_argument("--weight_decay_end", default=0.4)
        # net
        parser.add_argument("--dim", default=768)
        parser.add_argument("--m", default=1)
        parser.add_argument("--num_class", default=20)
        parser.add_argument("--VIT_pre_weight", default='vit.npz')
        self.parser = parser

    def parse(self):
        return self.parser.parse_args()


def main():
    args = Option().parse()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # <<<<<<<<<<<<<<<<datasets<<<<<<<<<<<<<<<<<<<<
    datasets = TrDataset(args.component_path, args.character_path, num_classes=args.num_class)
    train_loader = DataLoader(datasets, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)

    # <<<<<<<<<<<<<<<<models<<<<<<<<<<<<<<<<<<<<<<
    model_com = Encoder(args.num_class, checkpoint_path=args.VIT_pre_weight, feature_dim=args.dim).cuda()
    model_cha = Encoder(args.num_class, checkpoint_path=args.VIT_pre_weight, feature_dim=args.dim).cuda()

    # <<<<<<<<<<<<<<<<loss_initial<<<<<<<<<<<<<<<<
    loss_cn = torch.nn.CrossEntropyLoss().cuda()
    L_co_triplet = nn.TripletMarginLoss(margin=1, p=2).cuda()
    L_ch_triplet = losses.Varinace_loss(margin=args.m).cuda()

    # <<<<<<<<<<<<<<<<optimizer<<<<<<<<<<<<<<<<<<<<
    lr = args.lr
    weight_decay = args.weight_decay
    optimizer_com = torch.optim.AdamW(model_com.parameters(), lr, weight_decay=weight_decay)
    optimizer_cha = torch.optim.AdamW(model_cha.parameters(), lr, weight_decay=weight_decay)

    # <<<<<<<<<<<<<<<<scheduler initial<<<<<<<<<<<
    lr_schedule = utils.cosine_scheduler(
        lr * (args.batch_size) / 256.,  # linear scaling rule
        args.min_lr,
        args.epoch, len(train_loader),
        warmup_epochs=args.warmup_epochs,
        early_schedule_epochs=0,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,  # linear scaling rule
        args.weight_decay_end,
        args.epoch, len(train_loader),
        warmup_epochs=args.warmup_epochs,
        early_schedule_epochs=0,
    )
    fp16_scaler = torch.cuda.amp.GradScaler()

    min_loss = 1e5

    for epoch in range(args.epoch):
        epoch_train_loss = 0
        for batch_idx, data in enumerate(tqdm(train_loader)):

            it = len(train_loader) * epoch + batch_idx  # global training iteration
            for i, param_group in enumerate(optimizer_com.param_groups):
                if i == 0 or i == 1:
                    param_group['lr'] = lr_schedule[it] * 0.1
                else:
                    param_group["lr"] = lr_schedule[it]
                if i == 0 or i == 2:  # only the first group is regularized; look at get_params_groups for details
                    param_group["weight_decay"] = wd_schedule[it]
            for i, param_group in enumerate(optimizer_cha.param_groups):
                if i == 0 or i == 1:
                    param_group['lr'] = lr_schedule[it] * 0.1
                else:
                    param_group["lr"] = lr_schedule[it]
                if i == 0 or i == 2:  # only the first group is regularized; look at get_params_groups for details
                    param_group["weight_decay"] = wd_schedule[it]

            com, com_pos, cha, cha_neg, label = data

            com, com_pos, cha, cha_neg, label = com.cuda(), com_pos.cuda(), cha.cuda(), cha_neg.cuda(), label.cuda()

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                optimizer_com.zero_grad()
                optimizer_cha.zero_grad()

                com_logits, com_fea = model_com(com)  # class_logits,cls_token,
                com_pos_logits, com_pos_fea = model_com(com_pos)  # class_logits,cls_token,
                cha_logits, cha_fea = model_cha(cha)  # class_logits,cls_token,
                cha_neg_logits, cha_neg_fea = model_cha(cha_neg)  # class_logits,cls_token,

                cha_loss = L_ch_triplet(cha_fea, com_fea, com_pos_fea)
                com_loss = L_co_triplet(com_fea, cha_fea, cha_neg_fea)
                cn_loss = (loss_cn(com_logits, label) + loss_cn(cha_logits, label)) / 2

                loss = com_loss + cn_loss + cha_loss

                epoch_train_loss += loss.item()

                fp16_scaler.scale(loss).backward()
                fp16_scaler.step(optimizer_com)
                fp16_scaler.step(optimizer_cha)
                fp16_scaler.update()

        if epoch_train_loss / len(train_loader) < min_loss:
            torch.save(model_com.state_dict(), f'weights/com_weight.pth')
            torch.save(model_cha.state_dict(), f'weights/cha_weight.pth')
            min_loss = epoch_train_loss / len(train_loader)

        print('Epoch Train: [', epoch, '] Loss: ', epoch_train_loss, 'avg Loss: ',
              epoch_train_loss / len(train_loader))
        print(f'L_co_triplet:{com_loss} L_ch_triplet:{cha_loss} cls:{cn_loss}')


if __name__ == '__main__':
    main()
