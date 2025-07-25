"""
Author: ilikara 3435193369@qq.com
Date: 2025-07-25 13:39:02
LastEditors: ilikara 3435193369@qq.com
LastEditTime: 2025-07-25 15:14:49
FilePath: /nuedc2025-linux/main.py
Description: 多线程版本，摄像头采集和图像处理分离
"""

import time
import cv2
import threading
import subprocess
import os
import numpy as np
from queue import Queue
from red_line_tracker import find_red_segments_cv
from object_detect import ObjectDetector
from communicate import SerialPort

state = 0
state_depth = 0
dest_room = 0

def restart_camera_driver():
    # 使用 killall 终止所有 isp_media_server 进程
    try:
        subprocess.run(["killall", "isp_media_server"], check=True)
        print("已终止所有 isp_media_server 进程")
        time.sleep(1)  # 等待进程完全终止
    except subprocess.CalledProcessError:
        print("未找到运行的 isp_media_server 进程（或 killall 不可用）")

    # 重新启动 isp_media_server
    try:
        env = os.environ.copy()
        env["ISP_MEDIA_SENSOR_DRIVER"] = "/usr/lib/libvvcam.so"

        process = subprocess.Popen(
            ["/usr/bin/isp_media_server"],
            env=env,
            stdout=open("/dev/null", "w"),
            stderr=open("/tmp/isp.err.log", "a"),
        )
        print(f"已重新启动 isp_media_server，新进程 ID: {process.pid}")
        return True
    except Exception as e:
        print(f"启动 isp_media_server 时出错: {e}")
        return False


class CameraThread(threading.Thread):
    def __init__(self, camera_id="/dev/video1", width=640, height=360, fps=30):
        super().__init__()
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.frame_queue = Queue(maxsize=1)  # 只保留最新帧
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                # 如果队列已满，先丢弃旧帧
                if self.frame_queue.full():
                    self.frame_queue.get_nowait()
                self.frame_queue.put(frame)
            else:
                print("无法捕获图像")
                break
            time.sleep(0.01)  # 适当控制采集频率

    def stop(self):
        self.running = False
        self.cap.release()


class ProcessingThread(threading.Thread):
    def __init__(self, camera_thread, interval=0.05):
        super().__init__()
        self.camera_thread = camera_thread
        self.interval = interval
        self.detector = ObjectDetector(
            model_type="yolo",
            score_thresh=0.25,
            nms_thresh=0.5,
            debug_level=0,
            kmodel_path="./best.kmodel",
        )
        self.serial = SerialPort(serial_port="/dev/ttyS3", baudrate=115200)
        self.running = True

    def run(self):
        while self.running:
            start_time = time.time()

            # 从摄像头线程获取最新帧
            if not self.camera_thread.frame_queue.empty():
                frame = self.camera_thread.frame_queue.get()

                # 处理图像
                results = find_red_segments_cv(frame, [90, 180, 270])
                row = 180
                first_midpoint = 0
                if row in results and len(results[row]) > 0:
                    first_midpoint = results[row][0][1]  # 第一个片段的中点是索引1
                else:
                    continue
                print(f"{start_time}: mid: {first_midpoint}")
                self.serial.set_target("camera", first_midpoint - 320)
                self.serial.set_target("camera", first_midpoint - 320)

            # 计算执行时间并补偿延迟
            execution_time = time.time() - start_time
            sleep_time = max(0, self.interval - execution_time)
            time.sleep(sleep_time)

    def stop(self):
        self.serial.set_target("speed", 0)
        self.serial.set_target("speed", 0)
        self.serial.set_target("speed", 0)
        self.serial.set_target("speed", 0)
        self.running = False


if __name__ == "__main__":
    try:
        # 重启 ISP
        restart_camera_driver()
        # 创建并启动摄像头线程
        camera_thread = CameraThread()
        camera_thread.start()

        # 创建并启动处理线程
        processing_thread = ProcessingThread(camera_thread)
        processing_thread.serial.set_pid("motor", 1, 0.5, 0)
        processing_thread.serial.set_pid("motor", 1, 0.5, 0)
        # processing_thread.serial.set_pid("gyro", 0.2, 0.1, 0)
        # processing_thread.serial.set_pid("gyro", 0.2, 0.1, 0)
        processing_thread.serial.set_pid("camera", 1, 0, 1)
        processing_thread.serial.set_pid("camera", 1, 0, 1)
        processing_thread.serial.set_target("speed", 0)
        processing_thread.serial.set_target("speed", 0)

        if not camera_thread.frame_queue.empty():
            results = processing_thread.detector.detect_from_cv_image(
                camera_thread.frame_queue.get()
            )

        processing_thread.serial.set_target("speed", 500)
        processing_thread.serial.set_target("speed", 500)

        processing_thread.start()

        # 主线程等待
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("正在停止线程...")
        processing_thread.stop()
        camera_thread.stop()
        processing_thread.join()
        camera_thread.join()
        print("程序已退出")
