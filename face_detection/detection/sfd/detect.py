import torch
import torch.nn.functional as F

import os
import sys
import cv2
import random
import datetime
import math
import argparse
import numpy as np

import scipy.io as sio
import zipfile
from .net_s3fd import s3fd
from .bbox import *


def detect(net, img, device):
    img = img - np.array([104, 117, 123])  # 归一化操作[104, 117, 123]这些值分别对应 B（蓝色）、G（绿色）、R（红色）通道的平均像素值。减去均值后有助于快速收敛
    img = img.transpose(2, 0, 1)  # 改变通道顺序从 H×W×C 转换为 C×H×W。
    img = img.reshape((1,) + img.shape)  # 添加批次维度

    if 'cuda' in device:
        torch.backends.cudnn.benchmark = True  # 启动自动调优器，性能优化

    img = torch.from_numpy(img).float().to(device)  # 转化为图像张量
    BB, CC, HH, WW = img.size()
    with torch.no_grad():
        olist = net(img)

    bboxlist = []
    for i in range(len(olist) // 2):
        olist[i * 2] = F.softmax(olist[i * 2], dim=1)  # 对所有置信度分数Softmax
    olist = [oelem.data.cpu() for oelem in olist]  # 将模型输出从GPU转移到CPU
    for i in range(len(olist) // 2):
        ocls, oreg = olist[i * 2], olist[i * 2 + 1]  # 取出所有置信度分数和位置
        FB, FC, FH, FW = ocls.size()  # feature map size
        stride = 2**(i + 2)    # 步长4,8,16,32,64,128 步长定义了特征图上的一个像素点对应原始图像上的区域大小。随着网络深度的增加，特征图的尺寸变小，相应地步长变大。
        # 这样的设计允许网络在不同尺度上捕捉对象，较小的步长对应较大的特征图，能够检测小对象；较大的步长对应较小的特征图，适合检测大对象。
        anchor = stride * 4    # 基准大小 这种设置基于假设，即每个特征图层级的锚点应该覆盖比其步长更大的区域，以便捕捉到更大范围的潜在对象。
        poss = zip(*np.where(ocls[:, 1, :, :] > 0.05))  # 边界框候选点
        for Iindex, hindex, windex in poss:  # 深度，高度，宽度索引
            axc, ayc = stride / 2 + windex * stride, stride / 2 + hindex * stride  # 计算中心坐标，通过步长和索引映射出中心坐标
            score = ocls[0, 1, hindex, windex]  # 获取当前中心点的置信度分数，在二分类问题中1通常代表是（对象存在），0通常代表（对象不存在，背景）
            loc = oreg[0, :, hindex, windex].contiguous().view(1, 4)  # 获取当前中心点的坐标并重塑
            priors = torch.Tensor([[axc / 1.0, ayc / 1.0, stride * 4 / 1.0, stride * 4 / 1.0]])  # 创建先验框，通常大小为步长的四倍，足够大的先验框可以覆盖更大的区域，从而能够检测到更大的对象。
            variances = [0.1, 0.2]  # 定义方差
            box = decode(loc, priors, variances)  # 将位置偏移和先验框转换为实际的边界框坐标。
            x1, y1, x2, y2 = box[0] * 1.0  # 解码后的边界框坐标
            # cv2.rectangle(imgshow,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),1)
            bboxlist.append([x1, y1, x2, y2, score])
    bboxlist = np.array(bboxlist)
    if 0 == len(bboxlist):
        bboxlist = np.zeros((1, 5))  # 统一输出格式

    return bboxlist

def batch_detect(net, imgs, device):
    imgs = imgs - np.array([104, 117, 123])
    imgs = imgs.transpose(0, 3, 1, 2)

    if 'cuda' in device:
        torch.backends.cudnn.benchmark = True

    imgs = torch.from_numpy(imgs).float().to(device)  # (batch_size, channel, h, w)
    BB, CC, HH, WW = imgs.size()
    with torch.no_grad():
        olist = net(imgs)

    bboxlist = []
    for i in range(len(olist) // 2):
        olist[i * 2] = F.softmax(olist[i * 2], dim=1)
    olist = [oelem.data.cpu() for oelem in olist]
    for i in range(len(olist) // 2):
        ocls, oreg = olist[i * 2], olist[i * 2 + 1]
        FB, FC, FH, FW = ocls.size()  # feature map size
        stride = 2**(i + 2)    # 4,8,16,32,64,128
        anchor = stride * 4
        poss = zip(*np.where(ocls[:, 1, :, :] > 0.05))
        for Iindex, hindex, windex in poss:
            axc, ayc = stride / 2 + windex * stride, stride / 2 + hindex * stride
            score = ocls[:, 1, hindex, windex]
            loc = oreg[:, :, hindex, windex].contiguous().view(BB, 1, 4)
            priors = torch.Tensor([[axc / 1.0, ayc / 1.0, stride * 4 / 1.0, stride * 4 / 1.0]]).view(1, 1, 4)
            variances = [0.1, 0.2]
            box = batch_decode(loc, priors, variances)
            box = box[:, 0] * 1.0
            # cv2.rectangle(imgshow,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),1)
            bboxlist.append(torch.cat([box, score.unsqueeze(1)], 1).cpu().numpy())
    bboxlist = np.array(bboxlist)
    if 0 == len(bboxlist):
        bboxlist = np.zeros((1, BB, 5))

    return bboxlist
# 在图像被水平翻转后执行对象检测，并将检测到的边界框坐标转换回原始图像的坐标空间。
def flip_detect(net, img, device):
    img = cv2.flip(img, 1)  # 水平翻转
    b = detect(net, img, device)
    # 转换边界框坐标
    bboxlist = np.zeros(b.shape)
    bboxlist[:, 0] = img.shape[1] - b[:, 2]
    bboxlist[:, 1] = b[:, 1]
    bboxlist[:, 2] = img.shape[1] - b[:, 0]
    bboxlist[:, 3] = b[:, 3]
    bboxlist[:, 4] = b[:, 4]
    return bboxlist

# 将一组点（pts）转换成一个边界框
def pts_to_bb(pts):
    min_x, min_y = np.min(pts, axis=0)
    max_x, max_y = np.max(pts, axis=0)
    return np.array([min_x, min_y, max_x, max_y])
