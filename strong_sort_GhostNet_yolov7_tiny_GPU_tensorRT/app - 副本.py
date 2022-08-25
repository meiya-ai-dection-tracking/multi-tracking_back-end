import flask
from PIL import Image
from flask import Flask, json, request
from flask import jsonify
from flask_cors import *
# from strong_sort_GhostNet_yolov7_tiny_GPU_tensorRT1 import *
# from strong_sort_GhostNet_yolov7_tiny_GPU_tensorRT import demo
import demo
import objdetector
det = objdetector.Detector()
# from strong_sort_GhostNet_yolov7_tiny_CPU_tensorRT_j.demo import f
from werkzeug.datastructures import FileStorage
import numpy as np
import cv2
import json
import configparser
import os
import base64
from pymediainfo import MediaInfo
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
__PATH = {}

app = Flask(__name__)
# enable CORS
CORS(app, resource={r'/*': {'origins': '*'}})

socketio = SocketIO(app, cors_allowed_origins='*')
socketio.init_app(app, cors_allowed_origins='*')

name_space = '/test'
@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/trackByImg', methods=['POST'])
@cross_origin()
def track_by_img():
    # try:
        file = request.files.get("file")
        track_type = request.form['trackType'].split(',')
        file_type = request.form['fileType']
        file__ = request.form['suffix']
        print(file__)

        del_files('result')
        path = 'result\Test.' + file__
        file.save(path)
        if file is not None:
            if file_type == 'image':
                RESULT_PATH = demo.PICTURES(det, path, track_type)
                _PATH = RESULT_PATH
            elif file_type == 'video':
                RESULT_PATH = demo.VIDEO(det, path, track_type)
                global __PATH
                __PATH = RESULT_PATH
                print("__1",__PATH)
                _PATH = video_ffmpeg()

            return flask.send_file(_PATH,
                                   as_attachment=True)
        # else:
        #     return '上传文件失败'
    # except:
    #     # logger.exception('detect error')
    #     return 'detect error', 400

    # return jsonify({'detections': 'detections'})
@app.route('/pursuitTracking', methods=['GET'])
@cross_origin()
def Pursuit_tracking():
    global __PATH
    print("__2", __PATH)
    RESULT_PATH = encode(__PATH)
    return jsonify(RESULT_PATH)

@socketio.on('connect', namespace=name_space)
def connected_msg():
    emit('my response',{'data':'dakhbfkjabfquki'},name_space=name_space)
    print('client connected.')

@socketio.on('disconnect', namespace=name_space)
def disconnect_msg():
    print('client disconnected.')

@socketio.on('image', namespace=name_space)
def image(image):
    im = decode(image)
    # print(im)
    img = demo.CAPTURE(det,im,'person')
    # print(image,'\n')
    socketio.emit('image', {'data': 'img'}, namespace='image')
    # socketio.emit('image',{'data':'hello'}, namespace='image')

@socketio.on('test', namespace='api')   # 监听前端发回的包头 test ,应用命名空间为 api
def test():  # 此处可添加变量，接收从前端发回来的信息
    print('触发test函数')
    socketio.emit('api', {'data': 'test_OK'}, namespace='api') # 此处 api 对应前端 sockets 的 api

def decode(str):
    """
    img解码
    :param str:base64
    :return: img
    """
    if str != '':
        str = str.split(',')[1]
        img = base64.b64decode(str)
        encode_image = np.asarray(bytearray(img),dtype="uint8")
        im = cv2.imdecode(encode_image,cv2.IMREAD_COLOR)
        return im
def encode(RESULT_PATH):
    """
    img转码.
    :param RESULT_PATH:
    :return: RESULT_PATH
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
    视频转码
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
    del file
    :param dir_path:
    :return:
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
    socketio.run(app, host="172.20.10.2", debug=True, port=5000)


