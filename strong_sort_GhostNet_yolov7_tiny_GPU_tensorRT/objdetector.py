import time
import os
from strong_sort_GhostNet_yolov7_tiny_GPU_tensorRT.utils.datasets import letterbox
from strong_sort_GhostNet_yolov7_tiny_GPU_tensorRT.objtracker import update
from strong_sort_GhostNet_yolov7_tiny_GPU_tensorRT.YOLOv7_Tensorrt_master.infer import *

DETECTOR_PATH = os.path.dirname(__file__)+'\weights\yolov7_tiny.engine'

class baseDet(object):
    def __init__(self):
        self.img_size = 640
        self.threshold = 0.3
        self.stride = 1

    def build_config(self):
        self.frameCounter = 0

    def feedCap(self, im, obj_list):
        retDict = {
            'frame': None,
            'list_of_ids': None,
            'obj_bboxes': []
        }
        self.frameCounter += 1
        im, obj_bboxes = update(self, im, obj_list)
        retDict['frame'] = im
        retDict['obj_bboxes'] = obj_bboxes

        return retDict

    def init_model(self):
        raise EOFError("Undefined model type.")

    def preprocess(self):
        raise EOFError("Undefined model type.")

    def detect(self):
        raise EOFError("Undefined model type.")


class Detector(baseDet):
    """
        model init
    """
    def __init__(self):
        super(Detector, self).__init__()
        self.init_model()
        self.build_config()

    def init_model(self):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.m = TRT_engine(DETECTOR_PATH, self.device)
        self.names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    def preprocess(self, img):
        img0 = img.copy()
        img = letterbox(img, new_shape=self.img_size)[0]
        return img0, img

    def detect(self, im, obj_list):
        """

        :param im: numpy数据
        :param obj_list: 检测类型
        :return: {im0：检测前原图 pred_boxes：检测后含{{坐标}，属性，置信度}的列表 torch.tensor(cls_ids)：属性的torch格式}
        """
        t1=time.time()
        im0, pred = self.m.predict(im,threshold=0.5)
        print(time.time() - t1,'detect')
        pred_boxes = []
        cls_ids = []
        for det in pred:
            if det is not None and len(det):
                lbl, conf, x1, y1, x2, y2 = det
                if self.names[lbl] not in obj_list:
                    continue
                cls_ids.append(lbl)
                pred_boxes.append(
                        (x1, y1, x2, y2, lbl, conf))
        return im0, pred_boxes, torch.tensor(cls_ids)

