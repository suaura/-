from flask import Flask, request, jsonify, Response, render_template, url_for, send_file, session
from datetime import datetime
import torch
import time
import sys
import requests
import numpy as np
import pandas as pd
import os
import json
import cv2
import time
from PIL import Image
from camera import Camera
import webbrowser
import tracemalloc
from asyncio import streams
import subprocess
import psutil
from ultralytics import YOLO
from playsound import playsound


# YOLOv7 모델 로드
model = torch.hub.load(r'C:\Users/inkwabusan/PycharmProjects/pythonProject/yolov5', 'custom', 'yolov5/best.pt', source='local')


# 학습된 클래스 명
classes = ['0', '2', 'fire', 'smoke', 'spark']

key = 0

app = Flask(__name__)

c0 = cv2.VideoCapture(0)
c1 = cv2.VideoCapture(1)
c2 = cv2.VideoCapture(2)


# 이미지 저장을 위한 경로 설정
now = datetime.now()

siren_process = None
Sprinkler_process =None

# 캠 화면 웹 렌더링
@app.route('/')
def index():
    return render_template('index1.html')

# 구조요청
@app.route('/call_119')
def call_119():
    playsound("resque.mp3")
    return jsonify({'status': 'success'})


# 경보기.exe 실행
@app.route("/play_sound", methods=['GET', 'POST'])
def toggleSiren():
    global siren_process

    if siren_process is not None and siren_process.poll() is None:
        return jsonify({'status': 'failed', 'message': 'Siren is already playing'})

    siren_process = subprocess.Popen('dist/siren.exe')
    return jsonify({'status': 'success'})

@app.route("/play_Sprinkler", methods=['GET', 'POST'])
def Sprinkler():
    global Sprinkler_process

    if Sprinkler_process is not None and Sprinkler_process.poll() is None:
        return jsonify({'status': 'failed', 'message': 'Sprinkler is already playing'})

    Sprinkler_process = subprocess.Popen('dist/Sprinkler.exe')
    return jsonify({'status': 'success'})


@app.route("/stop_sound", methods=['GET', 'POST'])
def resetSiren():
    global key
    global cnt
    cap = globals()[f'c{cnt}']
    if request.method == 'POST':
        key = 0
        filename = f'static/no_fire/no_fire_{now.strftime("%Y-%m-%d_%H-%M-%S")}.jpg'
        save_image(filename, cap)
    return str(key)


def save_image(filename, cap):
    ret, frame = cap.read()
    cv2.imwrite(filename, frame)

def detect_objects():
    global key
    global cnt
    cnt = 0
    while True:
        if key == 0:
            cnt = (cnt + 1) % 3
            cap = globals()[f'c{cnt}']
        for i in range(100):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            results = model(frame)
            if cnt == 0:
                cv2.putText(frame, 'Engine room', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
                cv2.putText(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (270, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
            elif cnt == 1:
                cv2.putText(frame, 'Kitchen', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
                cv2.putText(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (270, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
            elif cnt == 2:
                cv2.putText(frame, 'Freight Area', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
                cv2.putText(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (270, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
            detections = results.xyxy[0]
            detected_classes = [classes[int(detection[5])] for detection in detections if detection[4] > 0.3]
            if any(cls in detected_classes for cls in classes):
                key = 1
                if cnt == 0:
                # 1캠 이미지 저장
                    cv2.imwrite('static/img/Engine room.jpg', frame)
                    cv2.putText(frame, 'Engine room', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
                    cv2.putText(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (270, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
                elif cnt == 1:
                # 2캠 이미지 저장
                    cv2.imwrite('static/img/Kitchen.jpg', frame)
                    cv2.putText(frame, 'Kitchen', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
                    cv2.putText(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (270, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
                elif cnt == 2:
                # 3캠 이미지 저장
                    cv2.imwrite('static/img/Freight Area.jpg', frame)
                    cv2.putText(frame, 'Freight Area', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
                    cv2.putText(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (270, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
                # 객체가 검출된 캠의 테두리 그리기
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)

            if '0' in detected_classes:
                for detection in detections:
                    if detection[4] > 0.3 and classes[int(detection[5])] == '0':
                        x1, y1, x2, y2 = map(int, detection[:4])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, 'fire', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            elif '2' in detected_classes:
                for detection in detections:
                    if detection[4] > 0.3 and classes[int(detection[5])] == '2':
                        x1, y1, x2, y2 = map(int, detection[:4])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, 'fire', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            elif 'fire' in detected_classes:
                for detection in detections:
                    if detection[4] > 0.3 and classes[int(detection[5])] == 'fire':
                        x1, y1, x2, y2 = map(int, detection[:4])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, 'fire', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            elif 'smoke' in detected_classes:
                for detection in detections:
                    if detection[4] > 0.3 and classes[int(detection[5])] == 'smoke':
                        x1, y1, x2, y2 = map(int, detection[:4])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, 'fire', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            elif 'spark' in detected_classes:
                for detection in detections:
                    if detection[4] > 0.3 and classes[int(detection[5])] == 'spark':
                        x1, y1, x2, y2 = map(int, detection[:4])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, 'fire', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)


            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



@app.route('/video_feed', methods=['GET', 'POST'])
def video_feed():
    return Response(detect_objects(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')




if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, threaded=True)