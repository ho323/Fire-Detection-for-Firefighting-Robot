import threading
import pyrealsense2 as rs
import numpy as np
import cv2
from flask import Flask, Response
import time

app = Flask(__name__)

# 글로벌 프레임 버퍼
latest_frame = None

def realsense_thread():
    global latest_frame
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            color_image = np.asanyarray(color_frame.get_data())
            latest_frame = color_image
    finally:
        pipeline.stop()

def gen_frames():
    while True:
        if latest_frame is None:
            continue
        ret, buffer = cv2.imencode('.jpg', latest_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.033)  # 약 30fps

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '<h1>D435 Live Stream</h1><img src="/video_feed">'

if __name__ == '__main__':
    t = threading.Thread(target=realsense_thread, daemon=True)
    t.start()
    app.run(host='0.0.0.0', port=5000, threaded=False, use_reloader=False)