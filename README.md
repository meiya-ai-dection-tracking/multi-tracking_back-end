# 多目标检测与跟踪后端项目 multi-tracking_back-end

## Introduction

多目标检测，即指出一张图片上每个目标的位置，并给出其具体类别与置信度。

多目标跟踪(Multi-Object Tracking, MOT)，就是对视频每一帧画面进行多目标检测的基础上，对每个目标分配一个ID，在目标运动过程中，维持每个目标的ID保持不变。

MOT的一般流程为：
1. 目标检测
2. 特征提取、运动预测 
3. 相似度计算 
4. 数据关联，其中检测器的好坏对于跟踪效果的影响很大。

MOT的研究重点更多在于相似度计算和数据关联方面，通过优秀的匹配算法可以让检测部分和跟踪部分相辅相成，提升性能。

此项目为后端项目，如需使用网页，需配合[前端项目](https://github.com/multi-object-tracking-meiya-ai/multi-tracking_front-end)使用

## Environment

* Tensorrt 8.4.1.5
* Cuda 10.2 Cudnn 8.4.1
* onnx 1.12.0
* Torch 1.7+（GPU运行为Torch 1.7.1+cu101）
* Torchvision 0.8+（GPU运行为Torchvision 0.8.2+cu101）
* opencv-python 4.6.0.66
* Flask 2.2.1 Flask-Cors 3.0.10 Flask-Socketio 5.2.0
* tensorrt 8.4.1.5

## File
**app.py**
* 常态初始化检测模型
* 与前端项目进行数据转换及处理
* 调用模型
  

**ffmpeg.exe**

使用时需要手动在[官网](https://ffmpeg.org/download.html)进行下载并添加到项目根目录

* 将返回前端的非’AVG‘编码格式的MP4输出视频转为‘H264’编码格式
    >ffmpeg.exe -i video.mp4 -vcodec h264 h264_result.mp4
    >>ffmpeg.exe -i video.mp4 -vcodec h264 -threads 5 -preset ultrafast h264_result.mp4

---
### **bytetrack_yolov7tiny_fast**
**main.py**
  * 实现视频、实时摄像头检测追踪的接口


**yolo/my_yolo.py**
  * 图像预处理
  * 非极大值抑制搜索局部最优解
  * 创建网络并载入模型
  * 检测边框

#### cython_bbox-0.1.3
* 下载解压[cython_bbox-0.1.3](https://pypi.org/project/cython-bbox/)
>打开解压文件夹下的set.py，更改extra_compile_args参数
> >extra_compile_args = {‘gcc’: [’/Qstd=c99’]}

* 更改后在解压文件夹下执行
>python setup.py build_ext install

* 若执行命令仍报错
>error: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/
> 
> 下载 [visualcppbuildtools full.exe](http://go.microsoft.com/fwlink/?LinkId=691126) 后运行exe文件

---
### **strong_sort_GhostNet_yolov7_tiny_GPU_tensorRT**
**demo.py**
  * 实现图像、视频检测追踪的接口


**objdetector.py**
  * 初始化engine模型
  * 数据预处理


**objtracker.py**
  * plot检测框以及目标检测点（追踪点）
  * 对每个第一次被检测的对象进行裁剪
  * 目标检测以及处理{(位置坐标),目标类型，置信度}进行裁剪和画框操作


#### YOLOv7_Tensorrt_master
* EfficientNMS.py和export_onnx.py复制到项目模型下，导出含有EfficientNMS的ONNX模型
>python export_onnx.py --weights ./weights/yolov7.pt

* 将生成的onnx模型复制到tensorrt/bin文件夹下，使用官方trtexec转化添加完EfficientNMS的onnx模型。FP32预测删除--fp16参数即可
>trtexec --onnx=./yolov7.onnx --saveEngine=./yolov7_fp16.engine --fp16 --workspace=200

* 等待生成序列化模型后，修改objdetector.py模型路径DETECTOR_PATH
>DETECTOR_PATH = os.path.dirname(__file__)+'\weights\yolov7_tiny.engine'

### flask
flask/app.py 本地网页实时摄像头
>run app.py

## Reference
https://github.com/WongKinYiu/yolov7

https://blog.csdn.net/PhilharmyWang/article/details/121612167

https://www.bilibili.com/video/BV1q34y1n7Bw?p=2

https://github.com/ifzhang/ByteTrack