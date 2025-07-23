'''
Author: ilikara 3435193369@qq.com
Date: 2025-07-20 08:51:27
LastEditors: ilikara 3435193369@qq.com
LastEditTime: 2025-07-23 12:37:07
FilePath: /nuedc2025-linux/saveimg.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import cv2
import time
import os

def capture_and_save_image(camera_index=0, output_dir='captured_images'):
    """
    从摄像头捕获图像并保存
    
    参数:
        camera_index: 摄像头索引(通常0是默认摄像头)
        output_dir: 保存图像的目录
    """
    # 创建输出目录(如果不存在)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 初始化摄像头
    cap = cv2.VideoCapture(camera_index)
    
    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_FPS, 30)
    # 捕获一帧图像
    ret, frame = cap.read()
    
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # 生成带时间戳的文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/capture_{timestamp}.jpg"
        
        # 保存图像
        cv2.imwrite(filename, frame)
        print(f"图像已保存为: {filename}")
    else:
        print("无法捕获图像")
    
    # 释放摄像头资源
    cap.release()

if __name__ == "__main__":
    # 对于树莓派摄像头模块，可能需要使用camera_index=0或camera_index=-1
    # 对于USB摄像头，通常使用camera_index=0
    capture_and_save_image(camera_index=1)