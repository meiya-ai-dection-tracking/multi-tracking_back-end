import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2 as cv
import time
import torch
import numpy as np
from bytetrack_yolov7tiny_fast.yolo.my_yolo import Yolo
from bytetrack_yolov7tiny_fast.yolo.utils.val_dataloader import LoadImages
from bytetrack_yolov7tiny_fast.yolo.utils.visualize import plot_tracking
from bytetrack_yolov7tiny_fast.tracker.byte_tracker import BYTETracker
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_PATH = os.path.dirname(__file__)+'/yolo/dataset/1.mp4'
image_size = (640,640)
RESULT_PATH = 'result//result.mp4'
import argparse
# 跟踪器相关参数
def parse_args():
    """
    init args
    :return: args
    """
    parser = argparse.ArgumentParser()
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.3, help="tracking confidence threshold")        # 0.5
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")         # 30
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")      # 0.8
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    args = parser.parse_args()
    return args

def VIDEO(yolo, path, args, results, tracker, track_type, DEVICE, frame_id=1):
    """
    视频目标检测及追踪
    :param yolo: 检测模型
    :param path: 前端返回视频路径
    :param args: 跟踪器参数
    :param results: 检测对象结果
    :param frame_id: 检测对象id
    :param tracker: 检测器模型
    :param track_type: 检测类型
    :param DEVICE: 环境
    """
    cap = cv.VideoCapture(path)
    fps = int(cap.get(5))
    videoWriter = None
    name = 'demo'
    t22 = time.time()
    while True:
        t1 = time.time()
        _, im0s = cap.read()

        if im0s is None:
            break
        frame_id += 1
        # print(time.time() - t1, 'read')
        t2 = time.time()
        outputs = yolo.detect_bounding_box(im0s,track_type, image_size, device=DEVICE)
        # print(time.time() - t2, 'det and post')
        if outputs is not None and len(outputs) > 1:
            # 跟新跟踪器
            online_targets = tracker.update(outputs, [im0s.shape[0], im0s.shape[1]], image_size)
            online_tlwhs = []
            online_ids = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )
            online_im = plot_tracking(
                im0s, online_tlwhs, online_ids, frame_id=frame_id, fps=1. / (time.time() - t1), e='VIDEO'
            )
            if videoWriter is None:
                fourcc = cv.VideoWriter_fourcc(
                    'm', 'p', '4', 'v')  # opencv3.0
                videoWriter = cv.VideoWriter(
                    RESULT_PATH, fourcc, fps, (online_im.shape[1], online_im.shape[0]))
            # print(time.time() - t1, 'all\n')
        else:
            online_im = im0s
        if online_im is not None and outputs is not None:
            videoWriter.write(online_im)
    # print(time.time()-t22)
    return online_im

def CAPTURE(yolo, im0s, args, results, frame_id, tracker, track_type):
    """
    实时摄像头检测
    :param yolo: 检测模型
    :param im0s: 视频帧图像
    :param args: 跟踪器参数
    :param results: 检测对象结果
    :param frame_id: 检测对象id
    :param tracker: 检测器模型
    :param track_type: 检测类型
    """
    t1 = time.time()

    if im0s is not None:
        outputs = yolo.detect_bounding_box(im0s,track_type, image_size, DEVICE)
        if outputs is not None and len(outputs) > 1:
            # 跟新跟踪器
            online_targets = tracker.update(outputs, [im0s.shape[0], im0s.shape[1]], image_size)
            online_tlwhs = []
            online_ids = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )
            online_im = plot_tracking(
                        im0s, online_tlwhs, online_ids, frame_id=frame_id, fps=1. / (time.time()-t1+0.001), e='CAPTURE'
                    )
        else:
            online_im = im0s
        return online_im
    return im0s

class bytetrack(object):
    """
        test
    """
    def __init__(self):
        # 载入YOLOX
        ALL_LIST = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                    'traffic light',
                    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                    'cow',
                    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
                    'frisbee',
                    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                    'surfboard',
                    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                    'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        DETECTOR_PATH = os.path.dirname(__file__) + '/yolo/weights/yolov7_tiny.pt'
        exp_file = [len(ALL_LIST), 0.001, 0.7]  # class num, conf_threshold, nms_threshold
        HALF = False  # model.half(), image.half()
        # 载入相关参数
        self.args = parse_args()
        # 载入YOLOX
        self.yolo = Yolo(exp_file, DETECTOR_PATH, ALL_LIST, device=DEVICE, half=HALF)
        # 载入跟踪器
        self.tracker = BYTETracker(self.args)




    def update(self, im0s, frame_id):
        t1 = time.time()
        obj_list = ['person']
        outputs = self.yolo.detect_bounding_box(im0s, obj_list,image_size, DEVICE)
        if outputs is not None and len(outputs) > 1:
            # 跟新跟踪器
            online_targets = self.tracker.update(outputs, [im0s.shape[0], im0s.shape[1]], image_size)
            online_tlwhs = []
            online_ids = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > self.args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
            if  time.time() - t1 < 0.001:
                fps = 64
            else:
                fps = 1.0/(time.time() - t1)
            online_im = plot_tracking(
                im0s, online_tlwhs, online_ids, frame_id=frame_id, fps=fps, e='VIDEO'
            )
        else:
            online_im = im0s
        return online_im

if __name__ == '__main__':
    from bytetrack_yolov7tiny_fast.yolo.my_yolo import Yolo
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 载入YOLOX
    ALL_LIST = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    DETECTOR_PATH = os.path.dirname(__file__) + '/yolo/weights/yolov7_tiny.pt'
    exp_file = [len(ALL_LIST), 0.001, 0.7]  # class num, conf_threshold, nms_threshold
    HALF = False  # model.half(), image.half()
    yolo = Yolo(exp_file, DETECTOR_PATH, ALL_LIST, device=DEVICE, half=HALF)
    args = parse_args()
    # 载入跟踪器
    tracker = BYTETracker(args)
    results = []
    frame_id = 0
    VIDEO(yolo, 'yolo/dataset/1.mp4', args, results, frame_id, tracker, ['person', 'car'], DEVICE)
