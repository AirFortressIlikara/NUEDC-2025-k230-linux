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


def run_detection(kmodel_path, score_thresh, nms_thresh, image_path, debug_level=0):
    """
    调用 C++ 检测程序并解析结果
    :param kmodel_path: 模型路径 (e.g., "yolov8n_640.kmodel")
    :param score_thresh: 分数阈值 (e.g., 0.3)
    :param nms_thresh: NMS 阈值 (e.g., 0.5)
    :param image_path: 图片路径 (e.g., "bus.jpg")
    :param debug_level: 调试级别 (0-2)
    :return: 解析后的检测结果列表
    """
    cmd = [
        "./ob_det.elf",
        kmodel_path,
        str(score_thresh),
        str(nms_thresh),
        image_path,
        str(debug_level),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error running C++ program:\n{result.stderr}")
        return []

    try:
        detections = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}\nRaw output:\n{result.stdout}")
        return []

    return detections


def save_results(image_path, detections, output_path="output.jpg"):
    """
    保存检测结果到图片（替代 imshow）
    :param image_path: 原始图片路径
    :param detections: 检测结果列表
    :param output_path: 保存路径 (默认 "output.jpg")
    """
    image = cv2.imread(image_path)
    for det in detections:
        box = det["box"]
        color = det["color"]
        cv2.rectangle(
            image,
            (box["x"], box["y"]),
            (box["x"] + box["width"], box["y"] + box["height"]),
            (int(color[0]), int(color[1]), int(color[2])),  # 确保颜色为整数
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

    # 保存图片（替换 imshow）
    cv2.imwrite(output_path, image)
    print(f"Results saved to: {output_path}")


def benchmark_detection(
    kmodel_path, score_thresh, nms_thresh, image_path, debug_level=0, runs=10
):
    """
    连续运行多次检测并统计耗时
    :param runs: 运行次数 (默认10次)
    :return: 平均耗时 (秒)
    """
    total_time = 0
    for i in range(runs):
        start_time = time.time()
        detections = run_detection(
            kmodel_path, score_thresh, nms_thresh, image_path, debug_level
        )
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


if __name__ == "__main__":
    # 示例调用
    kmodel = "yolov8n_640.kmodel"
    score_thresh = 0.3
    nms_thresh = 0.5
    image = "bus.jpg"
    debug_level = 0

    # 运行基准测试
    avg_time = benchmark_detection(kmodel, score_thresh, nms_thresh, image, debug_level)
