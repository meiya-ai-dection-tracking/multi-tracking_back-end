import cv2
from flask import Flask, render_template, Response

app = Flask(__name__)
from bytetrack_yolov7tiny_fast.main import bytetrack
b = bytetrack()
vid = cv2.VideoCapture(0)

@app.route('/')
def index():
    return render_template('index.html')

def real_time():
    id = 0
    while True:
        return_value, frame = vid.read()

        """byte track"""
        image = b.update(frame,id)

        image = cv2.imencode('.jpg', image)[1].tobytes()
        id += 1
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

@app.route('/video_byte_track')
def video_byte_track():
    return Response(real_time(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1212)

