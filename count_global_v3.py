import os
import cv2
import pandas as pd
from collections import defaultdict

# ============================================================
# Global ID 进出人数统计 V3
# ============================================================

print("=" * 60)
print("Global ID 进出人数统计 V3")
print("=" * 60)

# ------------------------------------------------------------
# 路径
# ------------------------------------------------------------
BASE_DIR = r"E:\people_flow"

CSV_PATH = os.path.join(
    BASE_DIR,
    "results",
    "trajectory_reconnect_v2",
    "reconnected_tracks.csv"
)

VIDEO_PATH = os.path.join(
    BASE_DIR,
    "videos",
    "test.mp4"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "count_global_v3"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "global_count_events.csv"
)

OUTPUT_VIDEO = os.path.join(
    OUTPUT_DIR,
    "global_count_result.mp4"
)

# ------------------------------------------------------------
# 参数
# ------------------------------------------------------------

# 计数线
LINE_Y = 360

# 滞回区域
TOP_LIMIT = 350
BOTTOM_LIMIT = 370

# 最少轨迹长度
MIN_TRACK_FRAMES = 10

# ============================================================
# 读取轨迹
# ============================================================

df = pd.read_csv(CSV_PATH)

print(f"轨迹数据量：{len(df)}")
print(f"Global ID 数量：{df['global_id'].nunique()}")

# ------------------------------------------------------------
# 删除过短 Global ID
# ------------------------------------------------------------

global_counts = df["global_id"].value_counts()

valid_ids = global_counts[
    global_counts >= MIN_TRACK_FRAMES
].index

df = df[df["global_id"].isin(valid_ids)].copy()

print(f"过滤后 Global ID：{df['global_id'].nunique()}")

# ------------------------------------------------------------
# 按帧排序
# ------------------------------------------------------------

df = df.sort_values(
    ["frame", "global_id"]
).reset_index(drop=True)

# ============================================================
# 区域判断
# ============================================================

def get_zone(y):

    if y < TOP_LIMIT:
        return "TOP"

    elif y > BOTTOM_LIMIT:
        return "BOTTOM"

    else:
        return "MIDDLE"


# ============================================================
# 为每个 Global ID 建立状态
# ============================================================

last_zone = {}

# 已经统计过的人员
counted_ids = set()

# 事件记录
events = []

IN_count = 0
OUT_count = 0

# ============================================================
# 逐帧处理
# ============================================================

for _, row in df.iterrows():

    global_id = int(row["global_id"])
    frame = int(row["frame"])
    time_sec = float(row["time"])

    cx = float(row["cx_smooth"])
    cy = float(row["cy_smooth"])

    current_zone = get_zone(cy)

    # 第一次出现
    if global_id not in last_zone:

        if current_zone != "MIDDLE":
            last_zone[global_id] = current_zone

        continue

    previous_zone = last_zone[global_id]

    # --------------------------------------------------------
    # MIDDLE 区域不改变状态
    # --------------------------------------------------------

    if current_zone == "MIDDLE":
        continue

    # --------------------------------------------------------
    # TOP -> BOTTOM
    # --------------------------------------------------------

    if (
        previous_zone == "TOP"
        and current_zone == "BOTTOM"
        and global_id not in counted_ids
    ):

        IN_count += 1
        counted_ids.add(global_id)

        events.append({
            "global_id": global_id,
            "direction": "IN",
            "frame": frame,
            "time": time_sec,
            "cx": cx,
            "cy": cy
        })

        print(
            f"IN  Global ID={global_id:4d} "
            f"time={time_sec:6.2f}s"
        )

    # --------------------------------------------------------
    # BOTTOM -> TOP
    # --------------------------------------------------------

    elif (
        previous_zone == "BOTTOM"
        and current_zone == "TOP"
        and global_id not in counted_ids
    ):

        OUT_count += 1
        counted_ids.add(global_id)

        events.append({
            "global_id": global_id,
            "direction": "OUT",
            "frame": frame,
            "time": time_sec,
            "cx": cx,
            "cy": cy
        })

        print(
            f"OUT Global ID={global_id:4d} "
            f"time={time_sec:6.2f}s"
        )

    # --------------------------------------------------------
    # 更新状态
    # --------------------------------------------------------

    last_zone[global_id] = current_zone


# ============================================================
# 保存统计结果
# ============================================================

events_df = pd.DataFrame(events)

if len(events_df) > 0:

    events_df = events_df.sort_values(
        "frame"
    ).reset_index(drop=True)

else:

    events_df = pd.DataFrame(
        columns=[
            "global_id",
            "direction",
            "frame",
            "time",
            "cx",
            "cy"
        ]
    )

events_df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# 输出统计结果
# ============================================================

print("\n")
print("=" * 60)
print("Global ID 进出统计完成")
print("=" * 60)

print(f"IN  人数：{IN_count}")
print(f"OUT 人数：{OUT_count}")
print(f"总人数：{IN_count + OUT_count}")

print("\n事件记录：")

if len(events_df) > 0:
    print(events_df.to_string(index=False))
else:
    print("没有检测到穿越计数线的事件")

print("\n事件文件：")
print(OUTPUT_CSV)

# ============================================================
# 生成结果视频
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():

    print("\n无法打开视频，跳过视频生成。")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

print("\n视频信息：")
print(f"分辨率：{width} × {height}")
print(f"FPS：{fps:.2f}")
print(f"总帧数：{total_frames}")

# ------------------------------------------------------------
# VideoWriter
# ------------------------------------------------------------

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)

# ------------------------------------------------------------
# 创建事件字典
# ------------------------------------------------------------

event_by_frame = defaultdict(list)

for _, event in events_df.iterrows():

    event_by_frame[int(event["frame"])].append(
        event
    )

# ============================================================
# 重新读取 Global ID 轨迹
# ============================================================

tracks_by_frame = defaultdict(list)

for _, row in df.iterrows():

    frame = int(row["frame"])

    tracks_by_frame[frame].append(row)

# ============================================================
# 视频绘制
# ============================================================

frame_index = 1

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # --------------------------------------------------------
    # 绘制计数线
    # --------------------------------------------------------

    cv2.line(
        frame,
        (0, LINE_Y),
        (width, LINE_Y),
        (255, 255, 0),
        3
    )

    # 上方确认线
    cv2.line(
        frame,
        (0, TOP_LIMIT),
        (width, TOP_LIMIT),
        (100, 100, 100),
        1
    )

    # 下方确认线
    cv2.line(
        frame,
        (0, BOTTOM_LIMIT),
        (width, BOTTOM_LIMIT),
        (100, 100, 100),
        1
    )

    # --------------------------------------------------------
    # 绘制 Global ID
    # --------------------------------------------------------

    if frame_index in tracks_by_frame:

        for row in tracks_by_frame[frame_index]:

            global_id = int(row["global_id"])

            cx = int(row["cx_smooth"])
            cy = int(row["cy_smooth"])

            # 根据 Global ID 生成颜色
            color = (
                (global_id * 37) % 255,
                (global_id * 67) % 255,
                (global_id * 97) % 255
            )

            # 当前中心点
            cv2.circle(
                frame,
                (cx, cy),
                6,
                color,
                -1
            )

            # Global ID
            cv2.putText(
                frame,
                f"GID:{global_id}",
                (cx + 8, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2
            )

    # --------------------------------------------------------
    # 当前帧事件
    # --------------------------------------------------------

    if frame_index in event_by_frame:

        for event in event_by_frame[frame_index]:

            direction = event["direction"]
            global_id = int(event["global_id"])

            text = (
                f"{direction}  "
                f"Global ID: {global_id}"
            )

            cv2.putText(
                frame,
                text,
                (width // 2 - 150, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                3
            )

    # --------------------------------------------------------
    # 左上角统计信息
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (10, 10),
        (260, 115),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"IN: {IN_count}",
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"OUT: {OUT_count}",
        (25, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 165, 255),
        2
    )

    cv2.putText(
        frame,
        f"TOTAL: {IN_count + OUT_count}",
        (25, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # 计数线文字
    # --------------------------------------------------------

    cv2.putText(
        frame,
        "COUNTING LINE",
        (20, LINE_Y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 0),
        2
    )

    writer.write(frame)

    frame_index += 1

# ============================================================
# 释放资源
# ============================================================

cap.release()
writer.release()

print("\n")
print("=" * 60)
print("结果视频生成完成")
print("=" * 60)

print(f"视频：{OUTPUT_VIDEO}")
print(f"事件：{OUTPUT_CSV}")

print("\n最终统计：")
print(f"IN  = {IN_count}")
print(f"OUT = {OUT_count}")
print(f"TOTAL = {IN_count + OUT_count}")

print("=" * 60)