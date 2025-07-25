import cv2
import numpy as np


def find_red_segments(image_path, rows, saturation_threshold=100):
    """
    从图像的指定行中查找红色部分的长度和中点

    参数:
        image_path: 图像路径
        rows: 要检查的行列表(如[100, 200, 300])
        saturation_threshold: 饱和度阈值(0-255)

    返回:
        字典，键为行号，值为该行红色片段的[起点, 中点, 终点, 长度]列表
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("无法加载图像，请检查路径")
    find_red_segments_cv(img, rows, saturation_threshold)


def find_red_segments_cv(img, rows, saturation_threshold=100):
    """
    从图像的指定行中查找红色部分的长度和中点

    参数:
        img: 图像
        rows: 要检查的行列表(如[100, 200, 300])
        saturation_threshold: 饱和度阈值(0-255)

    返回:
        字典，键为行号，值为该行红色片段的[起点, 中点, 终点, 长度]列表
    """
    # 转换为HSV色彩空间以便更好检测红色
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 定义红色范围 (HSV空间)
    lower_red1 = np.array([0, saturation_threshold, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, saturation_threshold, 50])
    upper_red2 = np.array([180, 255, 255])

    # 创建红色掩膜
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 | mask2

    results = {}

    for row in rows:
        if row >= img.shape[0]:
            print(f"警告: 行号 {row} 超出图像高度({img.shape[0]})")
            continue

        # 获取当前行的红色掩膜
        line_mask = red_mask[row, :]

        # 找到红色区域的起点和终点
        red_indices = np.where(line_mask > 0)[0]

        segments = []
        if len(red_indices) > 0:
            # 将连续的红色像素分组
            segments = np.split(red_indices, np.where(np.diff(red_indices) != 1)[0] + 1)

        segment_info = []
        for seg in segments:
            if len(seg) > 0:
                start = seg[0]
                end = seg[-1]
                length = end - start + 1
                midpoint = (start + end) // 2
                segment_info.append([start, midpoint, end, length])

        results[row] = segment_info

    return results


if __name__ == "__main__":
    image_path = "camera_output_6.jpg"  # 替换为你的图像路径
    rows_to_check = [90, 180, 270]  # 要检查的行号

    results = find_red_segments(image_path, rows_to_check)

    # 打印结果
    for row, segments in results.items():
        print(f"\n行 {row} 的红色片段:")
        for i, seg in enumerate(segments):
            print(
                f"  片段 {i+1}: 起点={seg[0]}, 中点={seg[1]}, 终点={seg[2]}, 长度={seg[3]}"
            )
