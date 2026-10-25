from ultralytics import YOLO
import cv2
import os
import csv

# ============================================================
# 1. 基础配置
# ============================================================

BASE_DIR = r"E:\people_flow"

MODEL_PATH = os.path.join(BASE_DIR, "yolo11n.pt")
VIDEO_PATH = os.path.join(BASE_DIR, "videos", "test.mp4")

OUTPUT_DIR = os.path.join(BASE_DIR, "results", "count_v6")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_VIDEO = os.path.join(
    OUTPUT_DIR,
    "count_result_v6.mp4"
)

EVENT_CSV = os.path.join(
    OUTPUT_DIR,
    "count_events_v6.csv"
)

# ============================================================
# 2. 计数参数
# ============================================================

# 计数线
LINE_Y = 360

# 滞回区域
# 上方确认区域
TOP_LIMIT = 350

# 下方确认区域
BOTTOM_LIMIT = 370

# 最少跟踪帧数
MIN_TRACK_FRAMES = 10

# 同一个 ID 两次计数的最小时间间隔
COOLDOWN = 2.0

# ============================================================
# 3. 加载模型
# ============================================================

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("无法打开视频：", VIDEO_PATH)
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# ============================================================
# 4. 输出视频
# ============================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)

# ============================================================
# 5. 轨迹状态
# ============================================================

# 每个 ID 的轨迹历史
track_history = {}

# 每个 ID 出现了多少帧
track_frames = {}

# 每个 ID 当前所处区域
track_zone = {}

# 上一次计数时间
last_count_time = {}

# 已经计数的 ID
counted_ids = set()

# ============================================================
# 6. 统计结果
# ============================================================

in_count = 0
out_count = 0

events = []

frame_idx = 0

# ============================================================
# 7. 判断区域
# ============================================================

def get_zone(y):

    if y < TOP_LIMIT:
        return "TOP"

    elif y > BOTTOM_LIMIT:
        return "BOTTOM"

    else:
        return "MIDDLE"


# ============================================================
# 8. 开始处理
# ============================================================

print("==============================")
print("人流计数系统 v6")
print("==============================")
print(f"分辨率：{width} × {height}")
print(f"FPS：{fps:.2f}")
print(f"总帧数：{total_frames}")
print(f"计数线 Y = {LINE_Y}")
print(f"上方确认线 = {TOP_LIMIT}")
print(f"下方确认线 = {BOTTOM_LIMIT}")
print("==============================")
print()
print("开始处理...")
print("--------------------------------")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_idx += 1

    current_time = frame_idx / fps

    # ========================================================
    # YOLO + ByteTrack
    # ========================================================

    results = model.track(
        frame,
        tracker="bytetrack.yaml",
        persist=True,
        classes=[0],
        conf=0.3,
        verbose=False
    )

    # ========================================================
    # 画计数线
    # ========================================================

    cv2.line(
        frame,
        (0, LINE_Y),
        (width, LINE_Y),
        (255, 255, 0),
        2
    )

    # 上边界
    cv2.line(
        frame,
        (0, TOP_LIMIT),
        (width, TOP_LIMIT),
        (0, 255, 255),
        1
    )

    # 下边界
    cv2.line(
        frame,
        (0, BOTTOM_LIMIT),
        (width, BOTTOM_LIMIT),
        (0, 255, 255),
        1
    )

    if len(results) == 0:
        writer.write(frame)
        continue

    result = results[0]

    if result.boxes is None or result.boxes.id is None:
        writer.write(frame)
        continue

    boxes = result.boxes.xyxy.cpu().numpy()
    ids = result.boxes.id.cpu().numpy().astype(int)

    # ========================================================
    # 遍历所有行人
    # ========================================================

    for box, track_id in zip(boxes, ids):

        x1, y1, x2, y2 = box

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # ----------------------------------------------------
        # 初始化 ID
        # ----------------------------------------------------

        if track_id not in track_history:

            track_history[track_id] = []

            track_frames[track_id] = 0

            track_zone[track_id] = get_zone(cy)

        # ----------------------------------------------------
        # 更新轨迹
        # ----------------------------------------------------

        track_history[track_id].append(
            (cx, cy)
        )

        # 只保存最近 100 个点
        if len(track_history[track_id]) > 100:
            track_history[track_id].pop(0)

        track_frames[track_id] += 1

        # ----------------------------------------------------
        # 当前区域
        # ----------------------------------------------------

        current_zone = get_zone(cy)

        previous_zone = track_zone[track_id]

        # ----------------------------------------------------
        # 满足最少跟踪帧数
        # ----------------------------------------------------

        if track_frames[track_id] >= MIN_TRACK_FRAMES:

            # =================================================
            # BOTTOM → TOP
            #
            # 离开
            # =================================================

            if (
                previous_zone == "BOTTOM"
                and current_zone == "TOP"
                and track_id not in counted_ids
            ):

                # 防止重复计数
                if (
                    track_id not in last_count_time
                    or
                    current_time -
                    last_count_time[track_id]
                    > COOLDOWN
                ):

                    out_count += 1

                    counted_ids.add(track_id)

                    last_count_time[track_id] = current_time

                    events.append(
                        {
                            "frame": frame_idx,
                            "time": round(current_time, 2),
                            "track_id": track_id,
                            "direction": "OUT",
                            "from": "BOTTOM",
                            "to": "TOP"
                        }
                    )

                    print(
                        f"[OUT] "
                        f"ID={track_id} "
                        f"time={current_time:.2f}s "
                        f"BOTTOM -> TOP"
                    )

            # =================================================
            # TOP → BOTTOM
            #
            # 进入
            # =================================================

            elif (
                previous_zone == "TOP"
                and current_zone == "BOTTOM"
                and track_id not in counted_ids
            ):

                if (
                    track_id not in last_count_time
                    or
                    current_time -
                    last_count_time[track_id]
                    > COOLDOWN
                ):

                    in_count += 1

                    counted_ids.add(track_id)

                    last_count_time[track_id] = current_time

                    events.append(
                        {
                            "frame": frame_idx,
                            "time": round(current_time, 2),
                            "track_id": track_id,
                            "direction": "IN",
                            "from": "TOP",
                            "to": "BOTTOM"
                        }
                    )

                    print(
                        f"[IN] "
                        f"ID={track_id} "
                        f"time={current_time:.2f}s "
                        f"TOP -> BOTTOM"
                    )

        # ----------------------------------------------------
        # 更新区域状态
        #
        # 注意：
        # MIDDLE 不覆盖 TOP/BOTTOM 状态
        # ----------------------------------------------------

        if current_zone != "MIDDLE":

            track_zone[track_id] = current_zone

        # ----------------------------------------------------
        # 画检测框
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            2
        )

        # ----------------------------------------------------
        # 画 ID
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"ID {track_id}",
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        # ----------------------------------------------------
        # 画中心点
        # ----------------------------------------------------

        cv2.circle(
            frame,
            (cx, cy),
            4,
            (0, 0, 255),
            -1
        )

        # ----------------------------------------------------
        # 画轨迹
        # ----------------------------------------------------

        points = track_history[track_id]

        for i in range(1, len(points)):

            cv2.line(
                frame,
                points[i - 1],
                points[i],
                (255, 0, 0),
                2
            )

    # ========================================================
    # 画统计信息
    # ========================================================

    cv2.rectangle(
        frame,
        (10, 10),
        (280, 100),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"IN : {in_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"OUT: {out_count}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    writer.write(frame)


# ============================================================
# 9. 保存事件 CSV
# ============================================================

with open(
    EVENT_CSV,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer_csv = csv.DictWriter(
        f,
        fieldnames=[
            "frame",
            "time",
            "track_id",
            "direction",
            "from",
            "to"
        ]
    )

    writer_csv.writeheader()

    writer_csv.writerows(events)


# ============================================================
# 10. 释放资源
# ============================================================

cap.release()
writer.release()

# ============================================================
# 11. 输出结果
# ============================================================

print()
print("================================")
print("人流计数完成")
print("================================")
print(f"进入人数：{in_count}")
print(f"离开人数：{out_count}")
print(f"总过线人数：{in_count + out_count}")
print()
print("事件文件：")
print(EVENT_CSV)
print()
print("结果视频：")
print(OUTPUT_VIDEO)
print("================================")