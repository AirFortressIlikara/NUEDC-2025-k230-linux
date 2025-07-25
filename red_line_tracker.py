import cv2
import numpy as np
import time


class RedLineTracker:
    def __init__(self, smooth_window=3, cross_range=20):
        self.smooth_window = smooth_window
        self.cross_range = cross_range

    def extract_red_mask(self, bgr_img):
        hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        return red_mask

    def find_track_points(self, mask, min_width_ratio=0.1, min_rows=2):
        height, width = mask.shape
        threshold_width = int(width * min_width_ratio)
        track = []
        cross_rows = []
        for y in range(height - 1, -1, -3):
            red_xs = [x for x in range(0, width, 3) if mask[y, x] > 0]
            avg_x = sum(red_xs) // len(red_xs) if red_xs else width // 2
            track.append([avg_x, y])

            if len(red_xs) >= threshold_width:
                cross_rows.append(y)

        if len(cross_rows) >= min_rows:
            # 取中间几行做平均计算中心点
            y_center = sum(cross_rows) // len(cross_rows)
            x_indices = np.where(mask[y_center] > 0)[0]
            if len(x_indices) > 0:
                x_center = int(np.mean(x_indices))
                return track, True, (x_center, y_center)

        return track, False, None

    def smooth_track(self, points):
        smoothed = []
        for i in range(len(points)):
            start = max(0, i - self.smooth_window)
            end = min(len(points), i + self.smooth_window + 1)
            sum_x = sum(p[0] for p in points[start:end])
            sum_y = sum(p[1] for p in points[start:end])
            count = end - start
            smoothed.append([sum_x // count, sum_y // count])
        return smoothed

    def draw_track(self, img, track, color=(0, 255, 0), thickness=2):
        for i in range(len(track) - 1):
            x0, y0 = track[i]
            x1, y1 = track[i + 1]
            for t in range(-thickness, thickness + 1):
                cv2.line(img, (x0 + t, y0), (x1 + t, y1), color, 1)

    def select_evenly_spaced_points(self, points, y_min, y_max, count=5):

        # 筛选处于 y_min ~ y_max 范围内的点
        candidates = [p for p in points if y_min <= p[1] <= y_max]

        # 如果候选点太少，直接返回全部
        if len(candidates) <= count:
            return candidates

        # 等间距选择
        step = len(candidates) / (count - 1)
        selected = [candidates[int(i * step)] for i in range(count - 1)]
        selected.append(candidates[-1])  # 保证最后一个是最大 y
        return selected

    def predict_ab_from_points(self, points):
        if len(points) < 2:
            return None  # 点太少无法拟合
        ys = np.array([p[1] for p in points])
        xs = np.array([p[0] for p in points])
        y_mean = np.mean(ys)
        x_mean = np.mean(xs)
        dy = ys - y_mean
        dx = xs - x_mean
        denom = np.sum(dy**2)
        if denom < 1e-6:
            return int(x_mean)  # y 值几乎相同，认为是水平线
        a = np.sum(dx * dy) / denom
        b = x_mean - a * y_mean
        return a, b

    def process(self, bgr_img, y_pred):
        mask = self.extract_red_mask(bgr_img)
        raw_track, found_cross, cross_center = self.find_track_points(mask)
        smoothed = self.smooth_track(raw_track)
        center_x = (
            smoothed[len(smoothed) // 2][0] if smoothed else bgr_img.shape[1] // 2
        )

        if found_cross:
            # 根据十字在画面中的高度决定用哪一边拟合
            y_threshold = cross_center[1]
            y_min = y_threshold - self.cross_range
            y_max = y_threshold + self.cross_range

            if y_threshold > bgr_img.shape[0] // 2:
                # 十字在图像靠下位置，选上半段拟合
                side_points = self.select_evenly_spaced_points(smoothed, 10, y_min, 5)
            else:
                side_points = self.select_evenly_spaced_points(
                    smoothed, y_max, bgr_img.shape[0] - 10, 5
                )

            a, b = self.predict_ab_from_points(side_points)

            # 只拟合前瞻处
            center_x = int(a * y_pred + b)

            # 拟合十字附近所有的点
            # for i in range(len(smoothed)):
            #     if y_min <= smoothed[i][1] <= y_max:
            #         smoothed[i][0] = int(a * smoothed[i][1] + b)

        # 绘制修改轨迹线
        # self.draw_track(bgr_img, smoothed, color=(0, 0, 255))

        return center_x, cross_center  # 中线， 十字坐标


if __name__ == "__main__":
    tracker = RedLineTracker(smooth_window=3)
    n = 1
    while True:
        n = n + 1
        print(n)
        ret, frame = tracker.cap.read()
        if ret:
            cx, cross_center = tracker.process(frame, 30)  # 30为前瞻
            if cross_center:
                print(f"检测到十字，中心位置: {cross_center}")
            else:
                print("未检测到十字")
            print(f"中线位置 x = {cx}")
        else:
            print("无法捕获图像")
            break
    del tracker
