import sys
import os
from pathlib import Path
import numpy as np
import cv2 as cv
from .models.experimental import attempt_load
import torchvision
import time
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """

    :param img:
    :param new_shape:
    :param color:
    :param auto:
    :param scaleFill:
    :param scaleup:
    :param stride:
    :return:
    """
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv.resize(img, new_unpad, interpolation=cv.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def preprocess(img, img_size, device, half=True):
    """
    预处理
    :param img: 图像
    :param img_size: 图像尺寸
    :param device: 环境
    :param half: 数据类型
    :return: 预处理后图像
    """
    img = letterbox(img, new_shape=img_size)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    if half:
        img = img.half()  # 精度
    else:
        img = img.float()  # 精度
    img /= 255.0  # 图像归一化
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img

# NMS算法
def postprocess(prediction, num_classes=1, conf_thre=0.7, nms_thre=0.45):
    """
    non maximum suppression 搜索局部极大值优化检测框
    :param prediction:检测器预测的边框、种类、置信度
    :param num_classes:网络模型载入参数
    :param conf_thre:检测目标置信度
    :param nms_thre:nms置信度
    :return:NMS算法后的检测器预测的边框、种类、置信度
    """
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(
            image_pred[:, 5: 5 + num_classes], 1, keepdim=True
        )
        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)



        detections = detections[conf_mask]

        if not detections.size(0):
            continue
        """0.015"""
        nms_out_index = torchvision.ops.batched_nms(
            detections[:, :4],
            detections[:, 4] * detections[:, 5],
            detections[:, 6],
            nms_thre,
        )
        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))
    return output


class Yolo(object):
    def __init__(self, exp_file, weight_path, all_list, device='gpu', half=True):
        # 创建网络并载入模型
        self.num_classes = exp_file[0]
        self.confthres = exp_file[1]
        self.nmsthre = exp_file[2]
        self.all_list = all_list

        self.half = half

        self.model = attempt_load(weight_path, map_location=device)
        self.model.to(device).eval()
        if self.half:
            self.model.half()      # CPU can not use half
        else:
            self.model.float()

    def detect_bounding_box(self, img0, obj_list, image_size, device):
        """
        检测边框
        :param img0:原图像
        :param obj_list:检测类型
        :param image_size:图像尺寸
        :param device:环境
        :return:检测器预测的边框、种类、置信度
        """
        img = preprocess(img0, image_size, device, half=self.half)
        self.obj_list = [self.all_list.index(i) for i in obj_list]
        with torch.no_grad():
            outputs = self.model(img)[0]
            # NMS
            outputs = postprocess(outputs, self.num_classes, self.confthres, self.nmsthre)[0]
            gain = min(img.shape[2] / img.shape[0], img.shape[3] / img0.shape[1])  # gain  = old / new
            pad = (img.shape[3] - img0.shape[1] * gain) / 2, (img.shape[2] - img0.shape[0] * gain) / 2  # wh padding

            if outputs is not None:
                mask = (outputs[:, 6] == self.obj_list[0]).squeeze()
                for i in range(len(self.obj_list) - 1):
                    mask = mask | (outputs[:, 6] == self.obj_list[i + 1]).squeeze()
                outputs = outputs[mask]
                if len(outputs.shape) == 3:
                    outputs = outputs.squeeze(0)

            if outputs is not None:
                outputs[:, [0, 2]] -= pad[0]
                outputs[:, [1, 3]] -= pad[1]
        return outputs