import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import strong_sort_GhostNet_yolov7_tiny_GPU_tensorRT.objtracker as objtracker
import strong_sort_GhostNet_yolov7_tiny_GPU_tensorRT.objdetector as objdetector
import imutils
import cv2
import torch

PICTURES_PATH =  os.path.dirname(__file__)+'//result//result.jpg'
VIDEO_TR_PERSON_PATH = os.path.dirname(__file__)+'//result//VIDEO_TR_PERSON//'
RESULT_PATH = 'result//result.mp4'
OBJ_LIST = 'person'
ALL_LIST = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
            'teddy bear', 'hair drier', 'toothbrush']
""" objtracker.person_boxes {pos_id:{[ndarray_pts]}} """

def drawtrack():

    # cv2.fillPoly(image, [ndarray_pts], color=(0, 0, 255))
    pass

def VIDEO(path, track_type):
    """
    视频检测追踪
    :param path: 前端返回的视频文件路径
    :param track_type: 检测类型
    :return: 视频中出现的每个id信息
    """
    print(track_type)
    print(torch.cuda.is_available())
    func_status = {}
    func_status['headpose'] = None

    name = 'demo'

    det = objdetector.Detector()
    cap = cv2.VideoCapture(path)
    fps = int(cap.get(5))
    t = int(1000 / fps)

    size = None
    videoWriter = None

    while True:
        # try:
        _, im = cap.read()
        if im is None:
            break
        t1 = time.time()
        result = det.feedCap(im, track_type)
        print(time.time() - t1, 'all\n')
        result = result['frame']
        result = imutils.resize(result, height=500)
        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc(
                'm', 'p', '4', 'v')  # opencv3.0
            videoWriter = cv2.VideoWriter(
                RESULT_PATH, fourcc, fps, (result.shape[1], result.shape[0]))

        videoWriter.write(result)
    _PATH = {}
    for key,value in objtracker.crop_img.items():
        _PATH[str(key)] = VIDEO_TR_PERSON_PATH + str(key) + '.jpg'
        im = cv2.imwrite(_PATH[str(key)], value)
    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()
    return _PATH

def PICTURES(file, obj_list):
    """
    图片检测追踪
    :param file: 前端返回检测图片保存路径
    :param obj_list: 检测类型
    :return: 检测追踪后结果路径
    """
    img = cv2.imread(file)
    det = objdetector.Detector()
    t1 = time.time()
    im, boxes, x = det.detect(img, obj_list)
    bboxes = []
    for (x1, y1, x2, y2, cls_id, pos_id) in boxes:
        bboxes.append((x1, y1, x2, y2, cls_id, round(pos_id, 2)))
    print(time.time() - t1, 'all')
    im = objtracker.plot_bboxes(im, bboxes, obj_list, type='picture')
    im = cv2.imwrite(PICTURES_PATH, im)

    return PICTURES_PATH

def CAPTURE(det, img, obj_list):
    """
    可用接口
    :param det: 检测模型
    :param img: 检测图片
    :param obj_list: 检测类型
    :return: 检测后图片
    """
    t1 = time.time()
    im, boxes, x = det.detect(img, obj_list)
    bboxes = []
    for (x1, y1, x2, y2, cls_id, pos_id) in boxes:
        bboxes.append((x1, y1, x2, y2, cls_id, round(pos_id, 2)))
    print(time.time() - t1, 'all')
    im = objtracker.plot_bboxes(im, bboxes, obj_list, type='CAPTURE')

    return im

file_path = ''
def VIDEOTRACK(pos_id):
    """
    IDX.txt:{帧率,坐标(x,y)}
    :param pos_id: 前端传回 pos_id : str -> int
    :return: 追踪划线视频路径 file_path : str
    """

    return file_path



if __name__ == '__main__':
    track_type = 'person'
    VIDEO('test_person.mp4', track_type)
