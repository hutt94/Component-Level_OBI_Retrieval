# -*- codeing = utf-8 -*-
# @Time : 2024/1/20 16:48
# @Author : 李昌杏
# @File : utils.py
# @Software : PyCharm
import datetime
import time
from collections import defaultdict
import numpy as np
import torch

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0, early_schedule_epochs=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    if early_schedule_epochs == 0:
        iters = np.arange(epochs * niter_per_ep - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        schedule = np.concatenate((warmup_schedule, schedule))
    else:
        iters = np.arange(early_schedule_epochs*niter_per_ep - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        remainder = np.array([final_value]*((epochs - early_schedule_epochs) * niter_per_ep))
        schedule = np.concatenate((warmup_schedule, schedule, remainder))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def cosine_scheduler_loss(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, early_schedule_epochs=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.array([base_value] * warmup_iters)

    if early_schedule_epochs == 0:
        iters = np.arange(epochs * niter_per_ep - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        schedule = np.concatenate((warmup_schedule, schedule))
    else:
        iters = np.arange(early_schedule_epochs * niter_per_ep - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        remainder = np.array([final_value]*((epochs - early_schedule_epochs) * niter_per_ep))
        schedule = np.concatenate((warmup_schedule, schedule, remainder))

    assert len(schedule) == epochs * niter_per_ep
    return schedule

def map_sake(predicted_features_gallery, gt_labels_gallery, predicted_features_query, gt_labels_query, scores, k=None):
    # mAP_ls = [[] for _ in range(len(np.unique(gt_labels_query)))]
    mean_mAP = []
    for fi in range(predicted_features_query.shape[0]):
        mapi = eval_AP_inner(gt_labels_query[fi], scores[fi], gt_labels_gallery, top=k)
        # mAP_ls[gt_labels_query[fi]].append(mapi)
        mean_mAP.append(mapi)
    # for i in range(len(mAP_ls)):
    #     mAP_ls[i] = np.mean(mAP_ls[i])
    # print("map for all classes: ", np.nanmean(mean_mAP))
    # print(mAP_ls)
    # print('top 10 maximal map: ', np.argsort(-np.array(mAP_ls)))
    return mean_mAP

def prec_sake(predicted_features_gallery, gt_labels_gallery, predicted_features_query, gt_labels_query, scores, k=None):
    # compute precision for two modalities
    # prec_ls = [[] for _ in range(len(np.unique(gt_labels_query)))]
    mean_prec = []
    for fi in range(predicted_features_query.shape[0]):
        prec = eval_precision(gt_labels_query[fi], scores[fi], gt_labels_gallery, top=k)
        # prec_ls[gt_labels_query[fi]].append(prec)
        mean_prec.append(prec)
    # print("precision for all samples: ", np.nanmean(mean_prec))
    return np.nanmean(mean_prec)

def eval_AP_inner(inst_id, scores, gt_labels, top=None):
    pos_flag = gt_labels == inst_id
    # print(pos_flag.shape)

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

    ap = VOCap(rec, prec)
    return ap

def VOCap(rec, prec):
    mrec = np.append(0, rec)  # put 0 in the first element
    mrec = np.append(mrec, 1)  # put 1 in the last element

    mpre = np.append(0, prec)
    mpre = np.append(mpre, 0)

    for ii in range(len(mpre) - 2, -1, -1):  # sort mpre, the smaller, the latter
        mpre[ii] = max(mpre[ii], mpre[ii + 1])

    msk = [i != j for i, j in zip(mrec[1:], mrec[0:-1])]
    ap = np.sum((mrec[1:][msk] - mrec[0:-1][msk]) * mpre[1:][msk])
    return ap

def eval_precision(inst_id, scores, gt_labels, top=100):
    pos_flag = gt_labels == inst_id
    tot = scores.shape[0]

    top = min(top, tot)

    sort_idx = np.argsort(-scores)
    return np.sum(pos_flag[sort_idx][:top]) / top

