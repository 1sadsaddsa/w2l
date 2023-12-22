from __future__ import print_function
import os
import sys
import cv2
import random
import datetime
import time
import math
import argparse
import numpy as np
import torch

try:
    from iou import IOU
except BaseException:
    # 计算两个矩形框的交并比，计算预测矩形框和真实矩形框的重叠程度
    def IOU(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
        sa = abs((ax2 - ax1) * (ay2 - ay1))
        sb = abs((bx2 - bx1) * (by2 - by1))
        x1, y1 = max(ax1, bx1), max(ay1, by1)
        x2, y2 = min(ax2, bx2), min(ay2, by2)
        w = x2 - x1
        h = y2 - y1
        if w < 0 or h < 0:
            return 0.0
        else:
            return 1.0 * w * h / (sa + sb - w * h)

# 计算边界框的位置和尺寸变换参数。这种变换通常用于对象检测任务中，以便将边界框的坐标转换为相对于某个参考框的尺度和位置的变化。
def bboxlog(x1, y1, x2, y2, axc, ayc, aww, ahh):
    xc, yc, ww, hh = (x2 + x1) / 2, (y2 + y1) / 2, x2 - x1, y2 - y1
    dx, dy = (xc - axc) / aww, (yc - ayc) / ahh
    dw, dh = math.log(ww / aww), math.log(hh / ahh)
    return dx, dy, dw, dh

# 它执行的是bboxlog函数的逆操作。这个函数用于将相对于参考框的位置和尺寸变化参数转换回绝对坐标的边界框。这在对象检测任务中用于从模型预测的相对变化中恢复到实际的边界框位置。
def bboxloginv(dx, dy, dw, dh, axc, ayc, aww, ahh):
    xc, yc = dx * aww + axc, dy * ahh + ayc
    ww, hh = math.exp(dw) * aww, math.exp(dh) * ahh
    x1, x2, y1, y2 = xc - ww / 2, xc + ww / 2, yc - hh / 2, yc + hh / 2
    return x1, y1, x2, y2

# NMS算法，用于去除重叠的边界框，保留最佳的检测结果
def nms(dets, thresh):
    if 0 == len(dets):
        return []
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)# 像素坐标需要+1
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # maximum逐元素比较i边界框的坐标和其他边界框坐标。得到重叠区域的左上角坐标和右下角坐标
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])

        w, h = np.maximum(0.0, xx2 - xx1 + 1), np.maximum(0.0, yy2 - yy1 + 1)
        ovr = w * h / (areas[i] + areas[order[1:]] - w * h)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

# 用于将标注的真实边界框（ground truth boxes）编码为相对于先验框（prior boxes）的偏移量。这在对象检测网络中非常常见，特别是在使用基于先验框（如 SSD）的检测方法时。
def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # dist b/t match center and prior's center 计算中心偏移
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    # encode variance 编码中心偏移
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh 计算尺寸比例，编码尺寸比例
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss 返回偏移和比例
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]

# 用于将对象检测模型的预测结果（通常是相对于先验框的偏移量）解码为绝对坐标的边界框。
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    # 计算中心坐标
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    # 左上角坐标
    boxes[:, :2] -= boxes[:, 2:] / 2
    # 右下角坐标
    boxes[:, 2:] += boxes[:, :2]
    return boxes
# 处理一批数据而非单个数据
def batch_decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :, :2] + loc[:, :, :2] * variances[0] * priors[:, :, 2:],
        priors[:, :, 2:] * torch.exp(loc[:, :, 2:] * variances[1])), 2)
    boxes[:, :, :2] -= boxes[:, :, 2:] / 2
    boxes[:, :, 2:] += boxes[:, :, :2]
    return boxes
