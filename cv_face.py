#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author William

# python cv_face.py haarcascade_frontalface_default.xml
# python cv_face.py --xml haarcascade_frontalface_default.xml

import cv2
import sys
import argparse

# cmd 另一种用法
# cascPath = sys.argv[1]

parser = argparse.ArgumentParser(description = 'face')
parser.add_argument('--xml', help = 'Path to xml file.',default='haarcascade_frontalface_default.xml')
args = parser.parse_args()

# 读取分级器文件
faceCascade = cv2.CascadeClassifier(args.xml)

# 打开摄像头
video_capture = cv2.VideoCapture(0)

while True:
    # 读取数据帧
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 可调参数
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.5,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # 框出脸的位置
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # 显示
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
video_capture.release()
cv2.destroyAllWindows()
