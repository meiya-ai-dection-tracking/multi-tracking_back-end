import time

import flask
import numpy
from PIL import Image
from flask import Flask, json, request
from flask import jsonify
from flask_cors import *
from strong_sort_GhostNet_yolov7_tiny_GPU_tensorRT import demo
from bytetrack_yolov7tiny_fast import main
from werkzeug.datastructures import FileStorage
import torch
import numpy as np
import cv2
import json
import configparser
import os
import base64
from io import BytesIO
from pymediainfo import MediaInfo
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from bytetrack_yolov7tiny_fast.yolo.my_yolo import Yolo
from bytetrack_yolov7tiny_fast.main import parse_args
from bytetrack_yolov7tiny_fast.tracker.byte_tracker import BYTETracker
import cv2
__PATH = {}
"""
    init model args
"""
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# 载入YOLOX
ALL_LIST = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
DETECTOR_PATH = os.path.dirname(__file__)+'/bytetrack_yolov7tiny_fast/yolo/weights/yolov7_tiny.pt'
exp_file = [len(ALL_LIST), 0.001, 0.7]      # class num, conf_threshold, nms_threshold
HALF = False    # model.half(), image.half()
yolo = Yolo(exp_file, DETECTOR_PATH, ALL_LIST, device=DEVICE, half=HALF)
# 载入相关参数
args = parse_args()
# 载入跟踪器
tracker = BYTETracker(args)
tracker_realtime = BYTETracker(args)
results = []
frame_id = 1
videoWriter = None
name = 'demo'
track_type_realtime = ''
app = Flask(__name__)
# enable CORS
CORS(app, supports_credentials=True, resource={r'/*': {'origins': '*'}})

socketio = SocketIO(app, cors_allowed_origins='*')
socketio.init_app(app, cors_allowed_origins='*')

name_space = '/test'
@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/trackByImg', methods=['POST'])
@cross_origin()
def track_by_img():
        """
        --检测图片--
        收到前端返回图片文件类型 调用检测
        :return:检测后结果图片
        """
    # try:
        file = request.files.get("file")
        track_type = request.form['trackType'].split(',')
        file__ = request.form['suffix']
        del_files('result')
        path = 'result\Test.' + file__
        file.save(path)
        if file is not None:
            global _PATH
            PICTURES_PATH = demo.PICTURES(path, track_type)
            _PATH = PICTURES_PATH
            return flask.send_file(_PATH,
                                   as_attachment=True)
@app.route('/trackByVideo', methods=['POST'])
@cross_origin()
def track_by_video():
    """
    --检测追踪视频--
    收到前端返回视频文件和检测模型 调用检测模型返回视频文件并解码
    :return:视频解码文件
    """
    file = request.files.get("file")
    track_type = request.form['trackType'].split(',')
    file__ = request.form['suffix']
    model = request.form['model']
    del_files('result')
    path = 'result\Test.' + file__
    file.save(path)
    print(model)
    if file is not None:
        global _PATH
        if int(model) == 1:
            t = time.time()
            exit = main.VIDEO(yolo, path, args, results, tracker, track_type, DEVICE, frame_id=1)
            _PATH = video_ffmpeg()
            print('Byte Track done...' + 'time is ' + str(time.time()-t))
        elif int(model) == 0:
            t = time.time()
            exit = demo.VIDEO(path, track_type)
            _PATH = video_ffmpeg()
            print('Strong Sort done...' + 'time is ' + str(time.time()-t))

        return flask.send_file(_PATH,
                                as_attachment=True)

@app.route('/pursuitTracking', methods=['GET'])
@cross_origin()
def Pursuit_tracking():
    global __PATH
    print("__2", __PATH)
    RESULT_PATH = encode(__PATH)
    return jsonify(RESULT_PATH)

@app.route('/realtimeTrackType', methods=['POST'])
@cross_origin()
def realtimeTrackType():
    """
    获取网页实时检测时返回的检测类型
    :return: 前端提示确认信息
    """
    global track_type_realtime
    track_type_realtime = request.form['trackType']
    return 'True'

@socketio.on('connect', namespace=name_space)
def connected_msg():
    emit('my response',{'connect':'dakhbfkjabfquki'},name_space='connect')
    print('client connected.')

@socketio.on('disconnect', namespace=name_space)
def disconnect_msg():
    print('client disconnected.')

@socketio.on('image', namespace=name_space)
def image(image):
    """
    --摄像头实时检测--
    实现对前端摄像头获取的视频帧读取并进行检测追踪
    :param image:前端返回视频帧流
    :return:base64格式下检测结果
    """
    # print(image, '\n')
    im = decode(image)
    img = main.CAPTURE(yolo, im, args, results, frame_id, tracker_realtime, [track_type_realtime])
    b64_code = cv2_base64(img)
    image_data = base64.b64decode(b64_code)
    emit('image', image_data, broadcast=True)

@socketio.on('test', namespace='api')   # 监听前端发回的包头 test ,应用命名空间为 api
def test():  # 此处可添加变量，接收从前端发回来的信息
    print('触发test函数')
    socketio.emit('api', {'data': 'test_OK'}, namespace='api') # 此处 api 对应前端 sockets 的 api

def cv2_base64(image):
    base64_str = cv2.imencode('.jpg',image)[1].tobytes()
    base64_str = base64.b64encode(base64_str)
    return base64_str

def decode(str):
    """
    img解码
    :param str:base64
    :return: cv2—img
    """
    if str != 'data:,':
        str0 = str.split(',')[1]
        img = base64.urlsafe_b64decode(str0)
        encode_image = np.asarray(bytearray(img),dtype="uint8")
        im = cv2.imdecode(encode_image,cv2.IMREAD_COLOR)
        if im is not None:
            return im
def encode(RESULT_PATH):
    """
    img转码.
    :param RESULT_PATH:img
    :return: RESULT_PATH：base64
    """
    for key, value in RESULT_PATH.items():
        if key != 0:
            im = cv2.imread(value)
            success, encoded_image = cv2.imencode(".jpg", im)
            # RESULT_PATH[key] = str(encoded_image.tobytes())
            RESULT_PATH[key] = base64.b64encode(encoded_image.read())
    print(RESULT_PATH)
    return RESULT_PATH
def video_ffmpeg():
    """
    --视频转码--
    使用ffmpeg.exe对视频编码格式进行转换
    :return: 转码后视频地址
    """
    print(MediaInfo.parse('result/result.mp4').to_data()['tracks'][1][
              'format'])
    if MediaInfo.parse('result/result.mp4').to_data()['tracks'][1][
        'format'] != 'AVC':
        cmd = "ffmpeg.exe" + " -i " + "result/result.mp4" + \
              " -vcodec h264 -threads 5 -preset ultrafast " + "result/h264_result.mp4"
        os.system(cmd)
    _PATH = 'result/h264_result.mp4'
    return _PATH
def del_files(dir_path):
    """
    --清空目标文件夹内所有文件--
    :param dir_path:目标文件夹路径
    """
    if os.path.isfile(dir_path):
        try:
            os.remove(dir_path)
        except BaseException as e:
            print(e)
    elif os.path.isdir(dir_path):
        file_lis = os.listdir(dir_path)
        for file_name in file_lis:
            tf = os.path.join(dir_path, file_name)
            del_files(tf)
if __name__ == '__main__':
    socketio.run(app, host="0.0.0.0", debug=True, port=5000)
    # app.run(host="0.0.0.0",port=5500)


