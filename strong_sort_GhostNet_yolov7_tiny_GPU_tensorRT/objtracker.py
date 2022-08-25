import time
from strong_sort_GhostNet_yolov7_tiny_GPU_tensorRT.deep_sort.utils.parser import get_config
from strong_sort_GhostNet_yolov7_tiny_GPU_tensorRT.deep_sort.strong_sort import DeepSort
import torch
import cv2
import numpy as np
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
ALL_LIST = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

cfg = get_config()

cfg.merge_from_file(os.path.dirname(__file__)+"/deep_sort/configs/deep_sort.yaml")
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    device= 'cuda' if torch.cuda.is_available() else 'cpu',
                    # device= 'cpu',
                    max_dist=cfg.DEEPSORT.MAX_DIST,
                    max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT,
                    nn_budget=cfg.DEEPSORT.NN_BUDGET)


def plot_bboxes(image, bboxes, obj_list, fps=0, type='video', line_thickness=None):
    """
    plot检测框以及目标检测点（追踪点）
    :param image: numpy image
    :param bboxes: {(位置坐标),目标类型，置信度}
    :param obj_list:检测类型
    :param fps:fps
    :param type:检测格式
    :param line_thickness:字体粗细
    :return: 含检测框、追踪点的图像
    """
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.001 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    tl_box = line_thickness or round(
        0.0015 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    list_pts = []
    point_radius = 4
    nums = len(obj_list)
    for (x1, y1, x2, y2, cls_id, pos_id) in bboxes:
        cls_id = ALL_LIST[int(cls_id)]
        if cls_id == obj_list[0]:
            color = (0,255,255)
        elif cls_id == obj_list[0]:
            color = (0,0,255)
        else:
            color = (255,0,255)

        # check whether hit line
        check_point_x = int(x1 + ((x2 - x1) * 0.5))
        check_point_y = int(y1 + ((y2 - y1) * 0.3))

        c1, c2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(image, c1, c2, color, thickness=tl_box, lineType=cv2.LINE_AA) # box
        tf = max(tl - 1, 1)  # font thickness
        if type == 'picture':
            cv2.putText(image, '{} conf-{}'.format(cls_id, pos_id), (c1[0], c1[1] - 2), 0, tl / 3,
                        [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        elif type == 'CAPTURE':
            cv2.putText(image, '{} conf-{}'.format(cls_id, pos_id), (c1[0], c1[1] - 2), 0, tl / 3,
                        [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
            cv2.putText(image, 'fps: %.2f num: %d' % (fps, nums),
                        (0, int(15 * 2)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)
        else:
            cv2.putText(image, '{} ID-{}'.format(cls_id, pos_id), (c1[0], c1[1] - 2), 0, tl / 3,
                        [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

        list_pts.append([check_point_x - point_radius, check_point_y - point_radius])
        list_pts.append([check_point_x - point_radius, check_point_y + point_radius])
        list_pts.append([check_point_x + point_radius, check_point_y + point_radius])
        list_pts.append([check_point_x + point_radius, check_point_y - point_radius])

        ndarray_pts = np.array(list_pts, np.int32)


        cv2.fillPoly(image, [ndarray_pts], color=(0, 0, 255))
        list_pts.clear()
    return image

person_boxes = {}
""" {pos_id:{[ndarray_pts]}} """
crop_img = {}
""" {pos_id:{img}} """
def cut_img(image,bboxes,obj_list):
    """
    对每个第一次被检测的对象进行裁剪
    :param image: 检测图像
    :param bboxes: {(位置坐标),目标类型，置信度}
    :param obj_list: 检测类型
    :return: 裁剪图像
    """
    for (x1, y1, x2, y2, cls_id, pos_id) in bboxes:
        cls_id = ALL_LIST[int(cls_id)]
        """
                       -------------track------------ 
        """
        if cls_id == 'person' and pos_id not in person_boxes:
            person_boxes[pos_id] = []
            crop_img[pos_id] = image[int(y1):int(y2), int(x1):int(x2)]
    return crop_img
def update(target_detector, image, obj_list):
        """
        目标检测以及处理{(位置坐标),目标类型，置信度}进行裁剪和画框操作
        :param target_detector: 模型
        :param image: 图像
        :param obj_list: 检测类型
        :return: image：检测后图像 bboxes2draw：{(位置坐标),目标类型，置信度}
        """
        im0, bboxes, cls_ids = target_detector.detect(image, obj_list)

        bbox_xywh = []
        confs = []
        bboxes2draw = []
        if len(bboxes):
            # Adapt detections to deep sort input format
            for x1, y1, x2, y2, _, conf in bboxes:
                obj = [
                    int((x1+x2)/2), int((y1+y2)/2),
                    x2-x1, y2-y1
                ]
                bbox_xywh.append(obj)
                confs.append(conf)
            xywhs = torch.Tensor(bbox_xywh)
            confss = torch.Tensor(confs)

            # Pass detections to deepsort
            outputs = deepsort.update(xywhs, confss, cls_ids, image)     # ——————不是track.output(tracker)
            for value in list(outputs):
                x1,y1,x2,y2,track_id,cls,_ = value
                bboxes2draw.append(
                    (x1, y1, x2, y2, cls, int(track_id))
                )
        cut_img(im0,bboxes2draw,obj_list)

        image = plot_bboxes(image, bboxes2draw, obj_list)
        return image, bboxes2draw
