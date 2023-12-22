import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2
from torch.utils.model_zoo import load_url
import common
from ..core import FaceDetector


from face_detection.detection import FaceDetector

HAS_CUDA = torch.cuda.is_available()
class HSigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class HSwish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            HSigmoid()
        )

    def forward(self, x):
        return x * self.se(self.pool(x))


class Block(nn.Module):
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class CBNModule(nn.Module):
    def __init__(self, inchannel, outchannel=24, kernel_size=3, stride=1, padding=0, bias=False):
        super(CBNModule, self).__init__()
        self.conv = nn.Conv2d(inchannel, outchannel, kernel_size, stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(outchannel)
        self.act = HSwish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class UpModule(nn.Module):
    def __init__(self, inchannel, outchannel=24, kernel_size=2, stride=2, bias=False):
        super(UpModule, self).__init__()
        self.dconv = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(inchannel, outchannel, 3, padding=1, bias=bias)
        self.bn = nn.BatchNorm2d(outchannel)
        self.act = HSwish()

    def forward(self, x):
        x = self.dconv(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class ContextModule(nn.Module):
    def __init__(self, inchannel):
        super(ContextModule, self).__init__()

        self.inconv = CBNModule(inchannel, inchannel, 3, 1, padding=1)

        half = inchannel // 2
        self.upconv = CBNModule(half, half, 3, 1, padding=1)
        self.downconv = CBNModule(half, half, 3, 1, padding=1)
        self.downconv2 = CBNModule(half, half, 3, 1, padding=1)

    def forward(self, x):
        x = self.inconv(x)
        up, down = torch.chunk(x, 2, dim=1)
        up = self.upconv(up)
        down = self.downconv(down)
        down = self.downconv2(down)
        return torch.cat([up, down], dim=1)


class DetectModule(nn.Module):
    def __init__(self, inchannel):
        super(DetectModule, self).__init__()

        self.upconv = CBNModule(inchannel, inchannel, 3, 1, padding=1)
        self.context = ContextModule(inchannel)

    def forward(self, x):
        up = self.upconv(x)
        down = self.context(x)
        return torch.cat([up, down], dim=1)


class DBFace(nn.Module):
    def __init__(self):
        super(DBFace, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = nn.ReLU(inplace=True)  # inplace=True原地修改数据

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1),  # 0
            Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, 2),  # 1
            Block(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1),  # 2
            Block(5, 24, 72, 40, nn.ReLU(inplace=True), SeModule(40), 2),  # 3
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),  # 4
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),  # 5
            Block(3, 40, 240, 80, HSwish(), None, 2),  # 6
            Block(3, 80, 200, 80, HSwish(), None, 1),  # 7
            Block(3, 80, 184, 80, HSwish(), None, 1),  # 8
            Block(3, 80, 184, 80, HSwish(), None, 1),  # 9
            Block(3, 80, 480, 112, HSwish(), SeModule(112), 1),  # 10
            Block(3, 112, 672, 112, HSwish(), SeModule(112), 1),  # 11
            Block(5, 112, 672, 160, HSwish(), SeModule(160), 1),  # 12
            Block(5, 160, 672, 160, HSwish(), SeModule(160), 2),  # 13
            Block(5, 160, 960, 160, HSwish(), SeModule(160), 1),  # 14
        )

        self.conv2 = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.hs2 = HSwish()

        self.conv3 = CBNModule(960, 320, kernel_size=1, stride=1, padding=0, bias=False)  # 32
        self.conv4 = CBNModule(320, 24, kernel_size=1, stride=1, padding=0, bias=False)  # 32
        self.conn0 = CBNModule(24, 24, 1, 1)  # s4
        self.conn1 = CBNModule(40, 24, 1, 1)  # s8
        self.conn3 = CBNModule(160, 24, 1, 1)  # s16

        self.up0 = UpModule(24, 24, 2, 2)  # s16
        self.up1 = UpModule(24, 24, 2, 2)  # s8
        # self.up2 = UpModule(24, 24, 2, 2)  # s4
        self.up2 = UpModule(24, 24, 2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((240, 135))
        self.cout = DetectModule(24)
        self.head_hm = nn.Conv2d(48, 1, 1)
        self.head_tlrb = nn.Conv2d(48, 1 * 4, 1)
        self.head_landmark = nn.Conv2d(48, 1 * 10, 1)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))

        keep = {"2": None, "5": None, "12": None}
        for index, item in enumerate(self.bneck):
            out = item(out)

            if str(index) in keep:
                keep[str(index)] = out

        out = self.hs2(self.bn2(self.conv2(out)))
        s32 = self.conv3(out)
        s32 = self.conv4(s32)
        s16 = self.up0(s32) + self.conn3(keep["12"])
        s8 = self.up1(s16) + self.conn1(keep["5"])
        s4 = self.adaptive_pool(self.up2(s8)) + self.conn0(keep["2"])    #  up2:torch.Size([1, 24, 240, 136]),con2:torch.Size([1, 24, 240, 135])
        out = self.cout(s4)

        hm = self.head_hm(out)  # 生成热图
        tlrb = self.head_tlrb(out)  # 生成边界框坐标
        landmark = self.head_landmark(out)  # 生成关键点坐标

        sigmoid_hm = hm.sigmoid()
        tlrb = torch.exp(tlrb)
        return sigmoid_hm, tlrb, landmark

    def load(self, file):
        print(f"load model: {file}")

        if torch.cuda.is_available():
            checkpoint = torch.load(file)
        else:
            checkpoint = torch.load(file, map_location=torch.device('cpu'))
        self.load_state_dict(checkpoint)

class DBFaceDetector(FaceDetector):
    def __init__(self, device, path_to_detector=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dbface.pth'), verbose=False):
        super(DBFaceDetector, self).__init__(device, verbose)

        # Initialise the face detector
        if not os.path.isfile(path_to_detector):
            raise FileNotFoundError(f"Model weight file not found at {path_to_detector}")
        else:
            model_weights = torch.load(path_to_detector, map_location=torch.device('cpu'))  # 其他人脸检测模型,默认使用GPU

        self.face_detector = DBFace()
        self.face_detector.load_state_dict(model_weights)
        self.face_detector.to(device)
        self.face_detector.eval()

    def detect_from_image(self, tensor_or_path):
        image = self.tensor_or_path_to_ndarray(tensor_or_path)  # 将图像张量或者路径转化为numpy数组

        bboxlist = detect(self.face_detector, image)
        keep = nms(bboxlist, 0.3)  # 非极大值限制，减少重叠度较高的检测框
        bboxlist = bboxlist[keep, :]  # 取出保留的边界框
        bboxlist = [x for x in bboxlist if x[-1] > 0.5]  # 保留置信度大于0.5的框

        return bboxlist

    def detect_from_batch(self, images):
        bboxlists = batch_detect(self.face_detector, images)
        return bboxlists

    @property
    def reference_scale(self):
        return 195

    @property
    def reference_x_shift(self):
        return 0

    @property
    def reference_y_shift(self):
        return 0

def nms(objs, iou=0.5):

    if objs is None or len(objs) <= 1:
        return objs

    objs = sorted(objs, key=lambda obj: obj.score, reverse=True)
    keep = []
    flags = [0] * len(objs)
    for index, obj in enumerate(objs):

        if flags[index] != 0:
            continue

        keep.append(obj)
        for j in range(index + 1, len(objs)):
            if flags[j] == 0 and obj.iou(objs[j]) > iou:
                flags[j] = 1
    return keep

def detect(model, image, threshold=0.4, nms_iou=0.5):

    mean = [0.408, 0.447, 0.47]
    std = [0.289, 0.274, 0.278]

    image = common.pad(image)
    image = ((image / 255.0 - mean) / std).astype(np.float32)
    image = image.transpose(2, 0, 1)

    torch_image = torch.from_numpy(image)[None]
    if HAS_CUDA:
        torch_image = torch_image.cuda()

    hm, box, landmark = model(torch_image)
    hm_pool = F.max_pool2d(hm, 3, 1, 1)
    scores, indices = ((hm == hm_pool).float() * hm).view(1, -1).cpu().topk(1000)
    hm_height, hm_width = hm.shape[2:]

    scores = scores.squeeze()
    indices = indices.squeeze()
    ys = list((indices / hm_width).int().data.numpy())
    xs = list((indices % hm_width).int().data.numpy())
    scores = list(scores.data.numpy())
    box = box.cpu().squeeze().data.numpy()
    landmark = landmark.cpu().squeeze().data.numpy()

    stride = 4

    bbox_list = []
    for cx, cy, score in zip(xs, ys, scores):
        if score < threshold:
            break

        x, y, r, b = box[:, cy, cx]
        xyrb = (np.array([cx, cy, cx, cy]) + [-x, -y, r, b]) * stride
        # 将边界框和置信度分数添加到列表中
        bbox_list.append([xyrb[0], xyrb[1], xyrb[2], xyrb[3], score])

    # 如果没有检测到任何对象，创建一个形状为 (1, 5) 的零数组
    if not bbox_list:
        return np.zeros((1, 5))

    # 将列表转换为 NumPy 数组
    return np.array(bbox_list)

def batch_detect(model, images, threshold=0.5, nms_iou=0.5):
    mean = np.array([0.408, 0.447, 0.47])
    std = np.array([0.289, 0.274, 0.278])

    # 处理图像批次
    images = images / 255.0
    images = (images - mean) / std
    images = images.transpose(0, 3, 1, 2).astype(np.float32)

    torch_images = torch.from_numpy(images)
    if HAS_CUDA:
        torch_images = torch_images.cuda()
    s = time.time()
    with torch.no_grad():
        hm, box, landmark = model(torch_images)
    e = time.time()
    print(e-s)
    batch_bbox_list = []
    for img_idx in range(torch_images.shape[0]):
        hm_img = hm[img_idx]
        box_img = box[img_idx]
        landmark_img = landmark[img_idx]

        # 后续处理与单图像相同，但针对每张图像分别处理
        hm_pool = F.max_pool2d(hm_img.unsqueeze(0), 3, 1, 1)  # 使用最大池化处理热图，帮助识别数据
        scores, indices = ((hm_img == hm_pool).float() * hm_img).view(1, -1).cpu().topk(10)  # topk(10)表示最多十个人脸
        # (hm_img == hm_pool).float() * hm_img)表示将hm_img中等于 hm_pool的值的元素保留下来
        # 去除大小为1的维度
        scores = scores.squeeze()
        indices = indices.squeeze()
        ys = list((indices % hm_img.shape[1]).int().data.numpy())  # 得到y坐标
        xs = list((indices / hm_img.shape[1]).int().data.numpy())  # 得到x坐标
        scores = list(scores.data.numpy())
        box_img = box_img.cpu().squeeze().data.numpy()
        landmark_img = landmark_img.cpu().squeeze().data.numpy()

        stride = 4
        bbox_list = []
        for cx, cy, score in zip(xs, ys, scores):
            if score < threshold:
                break
            x, y, r, b = box_img[:, cy, cx]  # r表示y的偏移量，b表示x的偏移量
            xyrb = (np.array([cx, cy, cx, cy]) + [-x, -y, r, b]) * stride  # 得到绝对位置坐标
            x5y5 = landmark_img[:, cy, cx]  # 提取关键点坐标
            x5y5 = (common.exp(x5y5 * 4) + ([cx]*5 + [cy]*5)) * stride  # 得到关键点的绝对坐标
            box_landmark = list(zip(x5y5[:5], x5y5[5:]))
            bbox_list.append([xyrb[0], xyrb[1], xyrb[2], xyrb[3], score])

        if not bbox_list:
            bbox_list = np.zeros((1, 5))

        batch_bbox_list.append(np.array(bbox_list))

    return batch_bbox_list

