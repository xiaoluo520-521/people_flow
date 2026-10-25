from ultralytics import YOLO
import cv2
import os
from collections import defaultdict

# ==============================
# 配置
# ==============================
VIDEO_PATH = r"E:\people_flow\videos\test.mp4"
MODEL_PATH = r"E:\people_flow\yolo11n.pt"

OUTPUT_DIR = r"E:\people_flow\results\debug"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_VIDEO = os.path.join(
    OUTPUT_DIR,
    "trajectory_debug.mp4"
)

CONF = 0.3

# ==============================
# 打开视频
# ==============================
cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

LINE_Y = height // 2

print()
print("==============================")
print("轨迹调试")
print("==============================")
print(f"分辨率：{width} × {height}")
print(f"FPS：{fps:.2f}")
print(f"总帧数：{frame_count}")
print(f"计数线 Y = {LINE_Y}")
print("==============================")
print()


# ==============================
# 输出视频
# ==============================
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)


# ==============================
# YOLO
# ==============================
model = YOLO(MODEL_PATH)


# ==============================
# 轨迹
# ==============================
track_history = defaultdict(list)

# 保存每个ID经过的最高/最低Y
track_min_y = defaultdict(lambda: 99999)
track_max_y = defaultdict(lambda: -99999)

# 是否已经打印过穿线信息
printed = set()


# ==============================
# 开始处理
# ==============================
frame_idx = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_idx += 1

    current_time = frame_idx / fps

    results = model.track(
        frame,
        tracker="bytetrack.yaml",
        persist=True,
        classes=[0],
        conf=CONF,
        verbose=False
    )

    result = results[0]

    # ==========================
    # 没有目标
    # ==========================
    if result.boxes.id is None:

        cv2.line(
            frame,
            (0, LINE_Y),
            (width, LINE_Y),
            (255, 255, 0),
            3
        )

        out.write(frame)
        continue

    boxes = result.boxes.xyxy.cpu().numpy()
    ids = result.boxes.id.int().cpu().tolist()

    # ==========================
    # 遍历目标
    # ==========================
    for box, track_id in zip(boxes, ids):

        x1, y1, x2, y2 = box

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # 保存轨迹
        track_history[track_id].append(
            (cx, cy)
        )

        if len(track_history[track_id]) > 100:
            track_history[track_id].pop(0)

        # 保存最大最小Y
        track_min_y[track_id] = min(
            track_min_y[track_id],
            cy
        )

        track_max_y[track_id] = max(
            track_max_y[track_id],
            cy
        )

        # ==========================
        # 检测是否穿过360
        # ==========================
        if (
            track_min_y[track_id] < LINE_Y
            and
            track_max_y[track_id] > LINE_Y
        ):

            if track_id not in printed:

                printed.add(track_id)

                print(
                    f"[CROSS] "
                    f"ID={track_id} "
                    f"time={current_time:.2f}s "
                    f"minY={track_min_y[track_id]} "
                    f"maxY={track_max_y[track_id]}"
                )

        # ==========================
        # 绘制检测框
        # ==========================
        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            2
        )

        cv2.circle(
            frame,
            (cx, cy),
            5,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            frame,
            f"ID:{track_id}",
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        # ==========================
        # 绘制轨迹
        # ==========================
        history = track_history[track_id]

        for i in range(1, len(history)):

            cv2.line(
                frame,
                history[i - 1],
                history[i],
                (255, 0, 255),
                2
            )

    # ==========================
    # 绘制计数线
    # ==========================
    cv2.line(
        frame,
        (0, LINE_Y),
        (width, LINE_Y),
        (255, 255, 0),
        3
    )

    # ==========================
    # 显示文字
    # ==========================
    cv2.putText(
        frame,
        f"LINE Y = {LINE_Y}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Frame: {frame_idx}",
        (30, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    out.write(frame)


# ==============================
# 释放
# ==============================
cap.release()
out.release()

print()
print("==============================")
print("调试完成")
print("==============================")
print("结果视频：")
print(OUTPUT_VIDEO)
print("==============================")