"""
Author: ilikara 3435193369@qq.com
Date: 2025-07-22 15:03:53
LastEditors: ilikara 3435193369@qq.com
LastEditTime: 2025-07-22 15:15:02
FilePath: /nuedc2025-linux/object_detect.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""

import json
import subprocess
import time
import cv2


class ObjectDetector:
    def __init__(self, model_type="cloud", debug_level=0, **kwargs):
        """
        初始化目标检测器

        :param model_type: 模型类型 ("yolo" 或 "cloud", 默认"cloud")
        :param debug_level: 调试级别 (0-2, 默认0)
        :param kwargs: 其他参数，根据model_type不同而不同:
                      - 当model_type="yolo"时:
                        kmodel_path: 模型路径
                        score_thresh: 分数阈值 (默认0.3)
                        nms_thresh: NMS阈值 (默认0.5)
                      - 当model_type="cloud"时: 无额外参数
        """
        self.model_type = model_type
        self.debug_level = debug_level

        if self.model_type == "yolo":
            self.kmodel_path = kwargs.get("kmodel_path")
            self.score_thresh = kwargs.get("score_thresh", 0.3)
            self.nms_thresh = kwargs.get("nms_thresh", 0.5)
        elif self.model_type == "cloud":
            self.kmodel_path = None
            self.score_thresh = None
            self.nms_thresh = None
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def run_detection(self, image_path):
        """
        调用 C++ 检测程序并解析结果
        :param image_path: 图片路径
        :return: 解析后的检测结果列表
        """
        if self.model_type == "yolo":
            cmd = [
                "./ob_det.elf",
                self.kmodel_path,
                str(self.score_thresh),
                str(self.nms_thresh),
                image_path,
                str(self.debug_level),
            ]
        elif self.model_type == "cloud":
            cmd = [
                "./detection.elf",
                "deploy_config.json",
                image_path,
                str(self.debug_level),
            ]
        print(cmd)
        result = subprocess.run(cmd, capture_output=True, text=True)

        print(result)

        if result.returncode != 0:
            print(f"Error running C++ program:\n{result.stderr}")
            return []

        try:
            detections = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}\nRaw output:\n{result.stdout}")
            return []

        return detections

    def detect_from_cv_image(self, cv_image, temp_image_path="temp_detect.jpg"):
        """
        从OpenCV图像数据进行目标检测
        :param cv_image: OpenCV图像数据
        :param temp_image_path: 临时保存图片的路径
        :return: 检测结果列表
        """
        # 保存OpenCV图像到临时文件
        cv2.imwrite(temp_image_path, cv_image)

        # 调用检测函数
        return self.run_detection(temp_image_path)

    @staticmethod
    def save_results(image_path, detections, output_path="output.jpg"):
        """
        保存检测结果到图片
        :param image_path: 原始图片路径
        :param detections: 检测结果列表
        :param output_path: 保存路径 (默认 "output.jpg")
        """
        image = cv2.imread(image_path)
        for det in detections:
            box = det["box"]
            cv2.rectangle(
                image,
                (box["x"], box["y"]),
                (box["width"], box["height"]),
                (int(255), int(0), int(0)),  # 确保颜色为整数
                2,
            )
            label = f"{det['className']} {det['confidence']:.2f}"
            cv2.putText(
                image,
                label,
                (box["x"], box["y"] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        # 保存图片
        cv2.imwrite(output_path, image)
        print(f"Results saved to: {output_path}")

    def benchmark_detection(self, image_path, runs=10):
        """
        连续运行多次检测并统计耗时
        :param image_path: 图片路径
        :param runs: 运行次数 (默认10次)
        :return: 平均耗时 (秒)
        """
        total_time = 0
        for i in range(runs):
            start_time = time.time()
            detections = self.run_detection(image_path)
            elapsed = time.time() - start_time
            total_time += elapsed

            print(f"Run {i + 1}/{runs}:")
            if detections:
                print(f"  Detected {len(detections)} objects in {elapsed:.4f} seconds")
                for det in detections:
                    print(
                        f"Class: {det['className']} (ID: {det['class_id']}), "
                        f"Confidence: {det['confidence']:.4f}, "
                        f"Box: [x={det['box']['x']}, y={det['box']['y']}, "
                        f"w={det['box']['width']}, h={det['box']['height']}]"
                    )
            else:
                print("  Detection failed!")
            print("-" * 40)

        avg_time = total_time / runs
        print(f"\nBenchmark completed (runs={runs}):")
        print(f"  Total time: {total_time:.4f} seconds")
        print(f"  Average time per run: {avg_time:.4f} seconds")
        return avg_time

    def benchmark_camera_detection(
        self,
        camera_device="/dev/video1",
        runs=10,
        temp_image_path="temp_camera.jpg",
        output_prefix="camera_output",
    ):
        """
        从摄像头读取帧进行基准测试
        :param camera_device: 摄像头设备路径 (默认 "/dev/video1")
        :param runs: 运行次数 (默认10次)
        :param temp_image_path: 临时保存图片的路径
        :param output_prefix: 输出结果图片前缀
        :return: 平均耗时 (秒)
        """
        cap = cv2.VideoCapture(camera_device)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_device}")
            return -1

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        cap.set(cv2.CAP_PROP_FPS, 30)

        total_time = 0
        successful_runs = 0

        try:
            for i in range(runs):
                # 读取摄像头帧
                ret, frame = cap.read()
                if not ret:
                    print(f"Run {i + 1}/{runs}: Failed to capture frame")
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # 执行检测
                start_time = time.time()
                detections = self.detect_from_cv_image(frame, temp_image_path)
                elapsed = time.time() - start_time

                # 处理结果
                print(f"Run {i + 1}/{runs}:")
                if detections:
                    successful_runs += 1
                    total_time += elapsed
                    print(
                        f"  Detected {len(detections)} objects in {elapsed:.4f} seconds"
                    )
                    for det in detections:
                        print(
                            f"Class: {det['className']} (ID: {det['class_id']}), "
                            f"Confidence: {det['confidence']:.4f}, "
                            f"Box: [x={det['box']['x']}, y={det['box']['y']}, "
                            f"w={det['box']['width']}, h={det['box']['height']}]"
                        )

                    # 保存带结果的图片
                    output_path = f"{output_prefix}_{i + 1}.jpg"
                    self.save_results(temp_image_path, detections, output_path)
                else:
                    print("  No objects detected!")
                print("-" * 40)

        finally:
            cap.release()

        if successful_runs > 0:
            avg_time = total_time / successful_runs
            print(
                f"\nCamera benchmark completed (successful runs={successful_runs}/{runs}):"
            )
            print(f"  Total time: {total_time:.4f} seconds")
            print(f"  Average time per run: {avg_time:.4f} seconds")
            return avg_time
        else:
            print("\nCamera benchmark failed - no successful runs")
            return -1


if __name__ == "__main__":
    # 示例用法
    # detector = ObjectDetector(
    #     model_type="cloud",
    #     debug_level=0,
    # )
    detector = ObjectDetector(
        model_type="yolo",
        score_thresh=0.25,
        nms_thresh=0.5,
        debug_level=0,
        kmodel_path="./best.kmodel",
    )

    # 图片检测示例
    detections = detector.run_detection("qwq.jpg")
    detector.save_results("qwq.jpg", detections, "qwq_out.jpg")
    print(detections)

    # 摄像头基准测试示例
    camera_benchmark = detector.benchmark_camera_detection(
        camera_device="/dev/video1", runs=10
    )
