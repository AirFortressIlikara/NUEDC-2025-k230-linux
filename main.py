"""
Author: ilikara 3435193369@qq.com
Date: 2025-07-25 13:39:02
LastEditors: ilikara 3435193369@qq.com
LastEditTime: 2025-07-25 15:14:49
FilePath: /nuedc2025-linux/main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""

import time
import cv2
from red_line_tracker import RedLineTracker
from object_detect import ObjectDetector
from communicate import SerialPort

state = 0
state_depth = 0
dest_room = 0

if __name__ == "__main__":
    cap = cv2.VideoCapture("/dev/video1")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_FPS, 30)

    tracker = RedLineTracker(smooth_window=3)
    detector = ObjectDetector(
        model_type="yolo",
        score_thresh=0.25,
        nms_thresh=0.5,
        debug_level=0,
        kmodel_path="./best.kmodel",
    )
    serial = SerialPort(serial_port="/dev/ttyS3", baudrate=115200)
    interval = 0.05
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            resized_frame = cv2.resize(frame, (160, 90))
            cx, cross_center = tracker.process(resized_frame, 30)  # 30为前瞻
            if cross_center:
                print(f"{start_time}: cross: {cross_center}")
            print(f"{start_time}: mid: {80 - cx}")
            serial.set_target("camera", 80 - cx)
        else:
            print("无法捕获图像")
            break

        # 计算执行时间并补偿延迟
        execution_time = time.time() - start_time
        sleep_time = max(0, interval - execution_time)
        # time.sleep(sleep_time)
    cap.release()
