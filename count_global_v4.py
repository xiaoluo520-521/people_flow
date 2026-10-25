import os
import cv2
import pandas as pd
from collections import defaultdict

# ============================================================
# Global ID 鲁棒进出人数统计 V4
# ============================================================

print("=" * 60)
print("Global ID 鲁棒进出人数统计 V4")
print("=" * 60)

# ============================================================
# 1. 路径设置
# ============================================================

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
    "count_global_v4"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "global_count_events_v4.csv"
)

OUTPUT_VIDEO = os.path.join(
    OUTPUT_DIR,
    "global_count_result_v4.mp4"
)

# ============================================================
# 2. 参数设置
# ============================================================

# ------------------------------------------------------------
# 主计数线
# ------------------------------------------------------------

LINE_Y = 360

# ------------------------------------------------------------
# 上下状态确认区域
# ------------------------------------------------------------

TOP_LIMIT = 350
BOTTOM_LIMIT = 370

# ------------------------------------------------------------
# 穿过计数线之后，需要继续移动的距离
# ------------------------------------------------------------

CONFIRM_DISTANCE = 30

# ------------------------------------------------------------
# 最小轨迹长度
# ------------------------------------------------------------

MIN_TRACK_FRAMES = 10

# ============================================================
# 3. 检查文件
# ============================================================

if not os.path.exists(CSV_PATH):

    print("\n错误：找不到轨迹文件：")
    print(CSV_PATH)
    exit()

if not os.path.exists(VIDEO_PATH):

    print("\n错误：找不到视频文件：")
    print(VIDEO_PATH)
    exit()

# ============================================================
# 4. 读取轨迹数据
# ============================================================

df = pd.read_csv(CSV_PATH)

print(f"轨迹数据量：{len(df)}")
print(f"Global ID 数量：{df['global_id'].nunique()}")

# ============================================================
# 5. 过滤过短轨迹
# ============================================================

global_counts = df["global_id"].value_counts()

valid_ids = global_counts[
    global_counts >= MIN_TRACK_FRAMES
].index

df = df[
    df["global_id"].isin(valid_ids)
].copy()

print(
    f"过滤后 Global ID："
    f"{df['global_id'].nunique()}"
)

# ============================================================
# 6. 检查平滑坐标
# ============================================================

if "cx_smooth" not in df.columns:

    print("\n错误：CSV 中没有 cx_smooth 列。")

    print("当前列：")
    print(df.columns.tolist())

    exit()

if "cy_smooth" not in df.columns:

    print("\n错误：CSV 中没有 cy_smooth 列。")

    print("当前列：")
    print(df.columns.tolist())

    exit()

# ============================================================
# 7. 按 Global ID 和帧排序
# ============================================================

df = df.sort_values(
    ["global_id", "frame"]
).reset_index(drop=True)

# ============================================================
# 8. 区域判断函数
# ============================================================

def get_zone(y):

    if y < TOP_LIMIT:

        return "TOP"

    elif y > BOTTOM_LIMIT:

        return "BOTTOM"

    else:

        return "MIDDLE"


# ============================================================
# 9. 构建每一个 Global ID 的完整轨迹
# ============================================================

tracks = {}

for gid, group in df.groupby("global_id"):

    group = group.sort_values("frame")

    frames = group["frame"].values
    times = group["time"].values
    xs = group["cx_smooth"].values
    ys = group["cy_smooth"].values

    tracks[int(gid)] = {
        "frame": frames,
        "time": times,
        "cx": xs,
        "cy": ys
    }

# ============================================================
# 10. 开始穿线分析
# ============================================================

events = []

IN_count = 0
OUT_count = 0

print("\n")
print("=" * 60)
print("开始分析 Global ID 穿线轨迹")
print("=" * 60)
print()

# ============================================================
# 11. 对每一个 Global ID 进行分析
# ============================================================

for gid, track in tracks.items():

    frames = track["frame"]
    times = track["time"]
    xs = track["cx"]
    ys = track["cy"]

    # --------------------------------------------------------
    # 找到人员第一次进入 TOP 或 BOTTOM 的位置
    # --------------------------------------------------------

    initial_zone = None
    initial_index = None

    for i, y in enumerate(ys):

        zone = get_zone(y)

        if zone == "TOP":

            initial_zone = "TOP"
            initial_index = i
            break

        elif zone == "BOTTOM":

            initial_zone = "BOTTOM"
            initial_index = i
            break

    # 整条轨迹都在 MIDDLE
    if initial_zone is None:

        continue

    # --------------------------------------------------------
    # 当前状态
    # --------------------------------------------------------

    state = initial_zone

    # --------------------------------------------------------
    # 是否已经穿过主计数线
    # --------------------------------------------------------

    crossed = False

    # --------------------------------------------------------
    # 穿线信息
    # --------------------------------------------------------

    cross_frame = None
    cross_time = None
    cross_x = None
    cross_y = None

    # ========================================================
    # 从第一次确认状态开始遍历
    # ========================================================

    for i in range(
        initial_index + 1,
        len(ys)
    ):

        y_prev = float(ys[i - 1])
        y_curr = float(ys[i])

        frame_curr = int(frames[i])
        time_curr = float(times[i])
        x_curr = float(xs[i])

        # ====================================================
        # TOP 状态
        # ====================================================

        if state == "TOP":

            # ------------------------------------------------
            # TOP → 穿过 360
            # ------------------------------------------------

            if (
                not crossed
                and y_prev < LINE_Y
                and y_curr >= LINE_Y
            ):

                crossed = True

                cross_frame = frame_curr
                cross_time = time_curr
                cross_x = x_curr
                cross_y = y_curr

                print(
                    f"检测到穿线："
                    f"GID={gid} "
                    f"方向候选=IN "
                    f"穿线时间={time_curr:.2f}s "
                    f"cy={y_curr:.2f}"
                )

            # ------------------------------------------------
            # 穿线之后继续向下移动
            # ------------------------------------------------

            if crossed:

                required_y = (
                    LINE_Y + CONFIRM_DISTANCE
                )

                if y_curr >= required_y:

                    distance = (
                        y_curr - LINE_Y
                    )

                    IN_count += 1

                    events.append({
                        "global_id": gid,
                        "direction": "IN",
                        "frame": frame_curr,
                        "time": time_curr,
                        "cx": x_curr,
                        "cy": y_curr,
                        "cross_frame": cross_frame,
                        "cross_time": cross_time,
                        "cross_x": cross_x,
                        "cross_y": cross_y,
                        "cross_distance": distance
                    })

                    print(
                        f"IN  Global ID={gid:4d} "
                        f"time={time_curr:6.2f}s "
                        f"distance={distance:6.1f}px"
                    )

                    # 当前 Global ID 已经完成计数
                    break

        # ====================================================
        # BOTTOM 状态
        # ====================================================

        elif state == "BOTTOM":

            # ------------------------------------------------
            # BOTTOM → 穿过 360
            # ------------------------------------------------

            if (
                not crossed
                and y_prev > LINE_Y
                and y_curr <= LINE_Y
            ):

                crossed = True

                cross_frame = frame_curr
                cross_time = time_curr
                cross_x = x_curr
                cross_y = y_curr

                print(
                    f"检测到穿线："
                    f"GID={gid} "
                    f"方向候选=OUT "
                    f"穿线时间={time_curr:.2f}s "
                    f"cy={y_curr:.2f}"
                )

            # ------------------------------------------------
            # 穿线之后继续向上移动
            # ------------------------------------------------

            if crossed:

                required_y = (
                    LINE_Y - CONFIRM_DISTANCE
                )

                if y_curr <= required_y:

                    distance = (
                        LINE_Y - y_curr
                    )

                    OUT_count += 1

                    events.append({
                        "global_id": gid,
                        "direction": "OUT",
                        "frame": frame_curr,
                        "time": time_curr,
                        "cx": x_curr,
                        "cy": y_curr,
                        "cross_frame": cross_frame,
                        "cross_time": cross_time,
                        "cross_x": cross_x,
                        "cross_y": cross_y,
                        "cross_distance": distance
                    })

                    print(
                        f"OUT Global ID={gid:4d} "
                        f"time={time_curr:6.2f}s "
                        f"distance={distance:6.1f}px"
                    )

                    # 当前 Global ID 已经完成计数
                    break

        # ====================================================
        # 未穿线时更新状态
        # ====================================================

        if not crossed:

            if y_curr < TOP_LIMIT:

                state = "TOP"

            elif y_curr > BOTTOM_LIMIT:

                state = "BOTTOM"


# ============================================================
# 12. 保存事件 CSV
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
            "cy",
            "cross_frame",
            "cross_time",
            "cross_x",
            "cross_y",
            "cross_distance"
        ]
    )

events_df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# 13. 输出统计结果
# ============================================================

print("\n")
print("=" * 60)
print("V4 鲁棒进出统计完成")
print("=" * 60)

print(
    f"IN  人数：{IN_count}"
)

print(
    f"OUT 人数：{OUT_count}"
)

print(
    f"总人数：{IN_count + OUT_count}"
)

print("\n事件记录：")

if len(events_df) > 0:

    print(
        events_df.to_string(
            index=False
        )
    )

else:

    print("没有检测到有效穿线事件")

print("\n事件文件：")
print(OUTPUT_CSV)

# ============================================================
# 14. 打开原视频
# ============================================================

cap = cv2.VideoCapture(
    VIDEO_PATH
)

if not cap.isOpened():

    print("\n无法打开视频。")
    exit()

fps = cap.get(
    cv2.CAP_PROP_FPS
)

if fps <= 0:

    fps = 30

width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

total_frames = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)

print("\n视频信息：")
print(
    f"分辨率：{width} × {height}"
)

print(
    f"FPS：{fps:.2f}"
)

print(
    f"总帧数：{total_frames}"
)

# ============================================================
# 15. 创建输出视频
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)

# ============================================================
# 16. 建立按帧索引的轨迹
# ============================================================

tracks_by_frame = defaultdict(list)

for _, row in df.iterrows():

    frame = int(
        row["frame"]
    )

    tracks_by_frame[frame].append(
        row
    )

# ============================================================
# 17. 建立按帧索引的事件
# ============================================================

event_by_frame = defaultdict(list)

for _, event in events_df.iterrows():

    frame = int(
        event["frame"]
    )

    event_by_frame[frame].append(
        event
    )

# ============================================================
# 18. 开始生成可视化视频
# ============================================================

frame_index = 1

while True:

    ret, frame = cap.read()

    if not ret:

        break

    # ========================================================
    # 绘制上方确认线
    # ========================================================

    cv2.line(
        frame,
        (0, TOP_LIMIT),
        (width, TOP_LIMIT),
        (150, 150, 150),
        1
    )

    # ========================================================
    # 绘制主计数线
    # ========================================================

    cv2.line(
        frame,
        (0, LINE_Y),
        (width, LINE_Y),
        (255, 255, 0),
        3
    )

    # ========================================================
    # 绘制下方确认线
    # ========================================================

    cv2.line(
        frame,
        (0, BOTTOM_LIMIT),
        (width, BOTTOM_LIMIT),
        (150, 150, 150),
        1
    )

    # ========================================================
    # 绘制 Global ID
    # ========================================================

    if frame_index in tracks_by_frame:

        for row in tracks_by_frame[
            frame_index
        ]:

            gid = int(
                row["global_id"]
            )

            cx = int(
                row["cx_smooth"]
            )

            cy = int(
                row["cy_smooth"]
            )

            # 根据 Global ID 生成固定颜色
            color = (
                (gid * 37) % 255,
                (gid * 67) % 255,
                (gid * 97) % 255
            )

            # ------------------------------------------------
            # 中心点
            # ------------------------------------------------

            cv2.circle(
                frame,
                (cx, cy),
                6,
                color,
                -1
            )

            # ------------------------------------------------
            # Global ID
            # ------------------------------------------------

            cv2.putText(
                frame,
                f"GID:{gid}",
                (cx + 8, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2
            )

    # ========================================================
    # 绘制进出事件
    # ========================================================

    if frame_index in event_by_frame:

        for event in event_by_frame[
            frame_index
        ]:

            direction = event[
                "direction"
            ]

            gid = int(
                event["global_id"]
            )

            text = (
                f"{direction}  "
                f"GID:{gid}"
            )

            cv2.putText(
                frame,
                text,
                (
                    width // 2 - 120,
                    100
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                3
            )

    # ========================================================
    # 左上角统计面板
    # ========================================================

    cv2.rectangle(
        frame,
        (10, 10),
        (290, 135),
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
        (25, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 165, 255),
        2
    )

    cv2.putText(
        frame,
        f"TOTAL: {IN_count + OUT_count}",
        (25, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # ========================================================
    # 计数线文字
    # ========================================================

    cv2.putText(
        frame,
        "COUNTING LINE",
        (20, LINE_Y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 0),
        2
    )

    # ========================================================
    # 输出帧
    # ========================================================

    writer.write(frame)

    frame_index += 1

# ============================================================
# 19. 释放资源
# ============================================================

cap.release()
writer.release()

# ============================================================
# 20. 最终结果
# ============================================================

print("\n")
print("=" * 60)
print("V4 结果视频生成完成")
print("=" * 60)

print(
    f"视频：{OUTPUT_VIDEO}"
)

print(
    f"事件：{OUTPUT_CSV}"
)

print("\n最终统计：")

print(
    f"IN  = {IN_count}"
)

print(
    f"OUT = {OUT_count}"
)

print(
    f"TOTAL = {IN_count + OUT_count}"
)

print("=" * 60)