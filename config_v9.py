import cv2
import os
import json
import tkinter as tk
from tkinter import messagebox


# ============================================================
# 1. 基础配置
# ============================================================

BASE_DIR = r"E:\people_flow"

VIDEO_PATH = os.path.join(
    BASE_DIR,
    "videos",
    "test.mp4"
)

CONFIG_PATH = os.path.join(
    BASE_DIR,
    "config.json"
)


# ============================================================
# 2. 全局变量
# ============================================================

counting_line = []

roi_polygon = []

drawing_mode = None

drawing_finished = False


# ============================================================
# 3. 鼠标回调
# ============================================================

def mouse_callback(
    event,
    x,
    y,
    flags,
    param
):

    global counting_line
    global roi_polygon

    global drawing_mode

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # --------------------------------------------------------
    # 计数线
    # --------------------------------------------------------

    if drawing_mode == "line":

        if len(counting_line) < 2:

            counting_line.append(
                [x, y]
            )

            print(
                f"计数线点：({x}, {y})"
            )

    # --------------------------------------------------------
    # ROI
    # --------------------------------------------------------

    elif drawing_mode == "roi":

        roi_polygon.append(
            [x, y]
        )

        print(
            f"ROI 点：({x}, {y})"
        )


# ============================================================
# 4. 通用绘制窗口
# ============================================================

def draw_configuration(
    video_path
):

    global counting_line
    global roi_polygon
    global drawing_mode

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        raise RuntimeError(
            "无法打开视频"
        )

    ret, frame = cap.read()

    cap.release()

    if not ret:

        raise RuntimeError(
            "无法读取视频第一帧"
        )

    # ========================================================
    # 第一阶段：计数线
    # ========================================================

    drawing_mode = "line"

    counting_line = []

    window_name = (
        "Configuration"
    )

    cv2.namedWindow(
        window_name
    )

    cv2.setMouseCallback(
        window_name,
        mouse_callback
    )

    print("\n======================================")
    print("第一步：绘制计数线")
    print("======================================")
    print("左键点击两个点")
    print("R：重新绘制")
    print("Enter：确认")
    print("======================================")

    while True:

        display = frame.copy()

        # ----------------------------------------------------
        # 计数线
        # ----------------------------------------------------

        if len(counting_line) >= 1:

            p1 = tuple(
                counting_line[0]
            )

            cv2.circle(
                display,
                p1,
                7,
                (0, 255, 0),
                -1
            )

        if len(counting_line) >= 2:

            p1 = tuple(
                counting_line[0]
            )

            p2 = tuple(
                counting_line[1]
            )

            cv2.line(
                display,
                p1,
                p2,
                (0, 255, 0),
                4
            )

            cv2.circle(
                display,
                p2,
                7,
                (0, 255, 0),
                -1
            )

        # ----------------------------------------------------
        # 提示
        # ----------------------------------------------------

        cv2.rectangle(
            display,
            (10, 10),
            (520, 75),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            display,
            "Click 2 points for counting line",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            "R: Reset    Enter: Confirm",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

        cv2.imshow(
            window_name,
            display
        )

        key = cv2.waitKey(30) & 0xFF

        # Enter
        if key == 13:

            if len(counting_line) == 2:

                break

        # R
        elif key == ord("r"):

            counting_line = []

        # ESC
        elif key == 27:

            cv2.destroyAllWindows()

            raise SystemExit(
                "用户取消配置"
            )

    # ========================================================
    # 第二阶段：ROI
    # ========================================================

    drawing_mode = "roi"

    roi_polygon = []

    print("\n======================================")
    print("第二步：绘制区域 ROI")
    print("======================================")
    print("左键：添加顶点")
    print("R：重新绘制")
    print("Enter：确认")
    print("======================================")

    while True:

        display = frame.copy()

        # ----------------------------------------------------
        # ROI 点
        # ----------------------------------------------------

        for i, point in enumerate(
            roi_polygon
        ):

            x, y = point

            cv2.circle(
                display,
                (x, y),
                6,
                (255, 0, 0),
                -1
            )

            cv2.putText(
                display,
                str(i + 1),
                (
                    x + 8,
                    y - 8
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 0, 0),
                2
            )

        # ----------------------------------------------------
        # ROI 边界
        # ----------------------------------------------------

        if len(
            roi_polygon
        ) >= 2:

            pts = [
                tuple(p)
                for p in roi_polygon
            ]

            for i in range(
                len(pts) - 1
            ):

                cv2.line(
                    display,
                    pts[i],
                    pts[i + 1],
                    (255, 0, 0),
                    3
                )

        # ----------------------------------------------------
        # ROI 填充
        # ----------------------------------------------------

        if len(
            roi_polygon
        ) >= 3:

            pts = np_array(
                roi_polygon
            )

            overlay = display.copy()

            cv2.fillPoly(
                overlay,
                [pts],
                (255, 0, 0)
            )

            display = cv2.addWeighted(
                overlay,
                0.25,
                display,
                0.75,
                0
            )

            cv2.polylines(
                display,
                [pts],
                True,
                (255, 0, 0),
                3
            )

        # ----------------------------------------------------
        # 提示
        # ----------------------------------------------------

        cv2.rectangle(
            display,
            (10, 10),
            (520, 75),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            display,
            "Click points for ROI polygon",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            "R: Reset    Enter: Confirm",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

        cv2.imshow(
            window_name,
            display
        )

        key = cv2.waitKey(30) & 0xFF

        if key == 13:

            if len(
                roi_polygon
            ) >= 3:

                break

        elif key == ord("r"):

            roi_polygon = []

        elif key == 27:

            cv2.destroyAllWindows()

            raise SystemExit(
                "用户取消配置"
            )

    cv2.destroyAllWindows()

    return (
        counting_line,
        roi_polygon
    )


# ============================================================
# numpy 简单替代
# ============================================================

def np_array(points):

    import numpy as np

    return np.array(
        points,
        dtype=np.int32
    )


# ============================================================
# 5. 保存配置
# ============================================================

def save_config(
    counting_line,
    roi_polygon
):

    config = {

        "video": VIDEO_PATH,

        "counting_line": {

            "p1": counting_line[0],

            "p2": counting_line[1]
        },

        "roi": roi_polygon,

        "parameters": {

            "min_track_frames": 10,

            "confirm_distance": 30,

            "min_movement_distance": 20,

            "max_frame_gap": 2,

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

    print("\n======================================")
    print("配置保存成功")
    print("======================================")

    print(
        f"配置文件：{CONFIG_PATH}"
    )

    print(
        json.dumps(
            config,
            ensure_ascii=False,
            indent=4
        )
    )


# ============================================================
# 6. 主程序
# ============================================================

def main():

    print("\n")
    print("==============================================")
    print("        人流分析系统配置工具 V9")
    print("==============================================")

    print(
        f"\n视频：{VIDEO_PATH}"
    )

    if not os.path.exists(
        VIDEO_PATH
    ):

        raise FileNotFoundError(
            f"视频不存在：{VIDEO_PATH}"
        )

    line, roi = draw_configuration(
        VIDEO_PATH
    )

    save_config(
        line,
        roi
    )

    print("\n配置完成。")
    print(
        "下一步可以使用 config.json "
        "进行正式分析。"
    )


if __name__ == "__main__":

    main()