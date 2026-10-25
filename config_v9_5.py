import cv2
import json
import os
import numpy as np


# ============================================================
# V9.5 可视化配置工具
# 功能：
# 1. 鼠标设置进出计数线
# 2. 鼠标设置 ROI 区域
# 3. 自动保存 config.json
# ============================================================

VIDEO_PATH = r"E:\people_flow\videos\test.mp4"
CONFIG_PATH = r"E:\people_flow\config.json"


# ============================================================
# 第一阶段：设置计数线
# ============================================================

line_points = []


def line_mouse_callback(event, x, y, flags, param):
    global line_points

    if event == cv2.EVENT_LBUTTONDOWN:

        if len(line_points) < 2:
            line_points.append((x, y))
            print(f"计数线点 {len(line_points)}：({x}, {y})")


# ============================================================
# 第二阶段：设置 ROI
# ============================================================

roi_points = []


def roi_mouse_callback(event, x, y, flags, param):
    global roi_points

    if event == cv2.EVENT_LBUTTONDOWN:
        roi_points.append((x, y))
        print(f"ROI点 {len(roi_points)}：({x}, {y})")


# ============================================================
# 读取第一帧
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("无法打开视频：")
    print(VIDEO_PATH)
    exit()

ret, frame = cap.read()

if not ret:
    print("无法读取视频第一帧")
    cap.release()
    exit()

cap.release()

original_frame = frame.copy()

print("=" * 70)
print("人流进出计数与区域热度分析系统 V9.5")
print("=" * 70)

print()
print("视频：", VIDEO_PATH)
print("分辨率：", frame.shape[1], "×", frame.shape[0])


# ============================================================
# 第一阶段：选择计数线
# ============================================================

print()
print("=" * 70)
print("第一阶段：设置进出计数线")
print("=" * 70)

print("操作说明：")
print("  鼠标左键：点击两个点")
print("  R：重新选择")
print("  Enter：确认")
print("  Esc：退出")
print()

cv2.namedWindow("设置计数线", cv2.WINDOW_NORMAL)
cv2.resizeWindow("设置计数线", 1280, 720)

cv2.setMouseCallback("设置计数线", line_mouse_callback)

while True:

    display = original_frame.copy()

    # 已选择的点
    for i, point in enumerate(line_points):

        cv2.circle(
            display,
            point,
            7,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            display,
            f"P{i + 1}",
            (point[0] + 10, point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    # 两点之间画线
    if len(line_points) == 2:

        cv2.line(
            display,
            line_points[0],
            line_points[1],
            (0, 255, 255),
            3
        )

    # 顶部提示
    cv2.rectangle(
        display,
        (0, 0),
        (1280, 65),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        display,
        "COUNTING LINE: Click 2 points | R: Reset | ENTER: Confirm | ESC: Exit",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("设置计数线", display)

    key = cv2.waitKey(30) & 0xFF

    # R：重新选择
    if key in [ord("r"), ord("R")]:

        line_points = []

        print("已清除计数线，请重新选择。")

    # Enter：确认
    elif key == 13:

        if len(line_points) == 2:

            print()
            print("计数线设置完成：")
            print(
                f"P1 = {line_points[0]}"
            )
            print(
                f"P2 = {line_points[1]}"
            )

            break

        else:

            print("必须选择两个点！")

    # ESC
    elif key == 27:

        cv2.destroyAllWindows()
        exit()

cv2.destroyWindow("设置计数线")


# ============================================================
# 第二阶段：设置 ROI
# ============================================================

print()
print("=" * 70)
print("第二阶段：设置 ROI 区域")
print("=" * 70)

print("操作说明：")
print("  鼠标左键：依次点击 ROI 顶点")
print("  R：重新选择")
print("  Enter：确认")
print("  Esc：退出")
print()

roi_points = []

cv2.namedWindow("设置ROI", cv2.WINDOW_NORMAL)
cv2.resizeWindow("设置ROI", 1280, 720)

cv2.setMouseCallback("设置ROI", roi_mouse_callback)

while True:

    display = original_frame.copy()

    # --------------------------------------------------------
    # 绘制 ROI
    # --------------------------------------------------------

    if len(roi_points) > 0:

        for i, point in enumerate(roi_points):

            cv2.circle(
                display,
                point,
                6,
                (255, 0, 0),
                -1
            )

            cv2.putText(
                display,
                f"P{i + 1}",
                (point[0] + 8, point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

    # 连接 ROI 点
    if len(roi_points) >= 2:

        for i in range(len(roi_points) - 1):

            cv2.line(
                display,
                roi_points[i],
                roi_points[i + 1],
                (255, 0, 0),
                2
            )

    # 至少三个点时闭合
    if len(roi_points) >= 3:

        cv2.line(
            display,
            roi_points[-1],
            roi_points[0],
            (255, 0, 0),
            2
        )

        # 半透明区域
        overlay = display.copy()

        polygon = np.array(
            roi_points,
            dtype=np.int32
        )

        cv2.fillPoly(
            overlay,
            [polygon],
            (255, 0, 0)
        )

        display = cv2.addWeighted(
            overlay,
            0.20,
            display,
            0.80,
            0
        )

        # 重新画边界
        cv2.polylines(
            display,
            [polygon],
            True,
            (255, 0, 0),
            2
        )

    # --------------------------------------------------------
    # 顶部提示
    # --------------------------------------------------------

    cv2.rectangle(
        display,
        (0, 0),
        (1280, 65),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        display,
        "ROI: Click points | R: Reset | ENTER: Confirm | ESC: Exit",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("设置ROI", display)

    key = cv2.waitKey(30) & 0xFF

    # R
    if key in [ord("r"), ord("R")]:

        roi_points = []

        print("已清除 ROI，请重新选择。")

    # Enter
    elif key == 13:

        if len(roi_points) >= 3:

            print()
            print("ROI 设置完成：")

            for i, point in enumerate(roi_points):

                print(
                    f"P{i + 1} = {point}"
                )

            break

        else:

            print("ROI 至少需要 3 个点！")

    # ESC
    elif key == 27:

        cv2.destroyAllWindows()
        exit()


cv2.destroyWindow("设置ROI")


# ============================================================
# 保存配置
# ============================================================

config = {

    "video": VIDEO_PATH,

    "counting_line": {

        "p1": list(line_points[0]),
        "p2": list(line_points[1])
    },

    "roi": [

        list(point)
        for point in roi_points
    ],

    "parameters": {

        "min_track_frames": 10,

        "confirm_distance": 20,

        "min_movement_distance": 10,

        "max_frame_gap": 10,

        "heatmap_blur_kernel": 51
    }
}


with open(
    CONFIG_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        config,
        f,
        ensure_ascii=False,
        indent=4
    )


# ============================================================
# 最终显示
# ============================================================

print()
print("=" * 70)
print("V9.5 配置完成")
print("=" * 70)

print()
print("计数线：")
print(
    f"{line_points[0]} -> {line_points[1]}"
)

print()
print("ROI：")

for point in roi_points:
    print(point)

print()
print("参数：")
print("  min_track_frames      =", 10)
print("  confirm_distance      =", 20)
print("  min_movement_distance =", 10)
print("  max_frame_gap         =", 10)
print("  heatmap_blur_kernel   =", 51)

print()
print("配置文件已经保存：")
print(CONFIG_PATH)

print("=" * 70)