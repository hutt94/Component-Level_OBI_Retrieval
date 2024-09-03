# -*- codeing = utf-8 -*-
# @Time : 2024/1/20 17:07
# @Author : 李昌杏
# @File : losses.py
# @Software : PyCharm
import torch
import torch.nn as nn

class Varinace_loss(nn.Module):
    def __init__(self,margin=1):
        super(Varinace_loss, self).__init__()
        self.margin=margin

    def forward(self,a,p1,p2):
        dist1=torch.sqrt(torch.sum(torch.square(a-p1),dim=1))
        dist2=torch.sqrt(torch.sum(torch.square(a-p2),dim=1))

        basic_loss=torch.abs(dist1-dist2)-self.margin
        basic_loss=basic_loss.unsqueeze(dim=1)
        loss=torch.cat((basic_loss, torch.zeros_like(basic_loss)),dim=1).max(dim=1)
        return loss[0].mean()
