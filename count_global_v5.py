import os
import cv2
import math
import pandas as pd
from collections import defaultdict

# ============================================================
# Global ID 任意方向计数线统计 V5
# ============================================================

print("=" * 70)
print("Global ID 任意方向计数线统计 V5")
print("=" * 70)


# ============================================================
# 1. 路径
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
    "count_global_v5"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "global_count_events_v5.csv"
)

OUTPUT_VIDEO = os.path.join(
    OUTPUT_DIR,
    "global_count_result_v5.mp4"
)


# ============================================================
# 2. 参数
# ============================================================

# ------------------------------------------------------------
# 自定义计数线
#
# P1 ---------------- P2
#
# 你以后只需要修改这两个点
# ------------------------------------------------------------

P1 = (200, 360)
P2 = (1100, 360)

# ------------------------------------------------------------
# 最少轨迹长度
# ------------------------------------------------------------

MIN_TRACK_FRAMES = 10

# ------------------------------------------------------------
# 穿线之后，需要离开计数线多少像素才确认
# ------------------------------------------------------------

CONFIRM_DISTANCE = 30

# ------------------------------------------------------------
# 最小运动距离
#
# 防止人员在计数线附近轻微抖动
# ------------------------------------------------------------

MIN_MOVEMENT_DISTANCE = 20

# ------------------------------------------------------------
# 是否保存轨迹调试信息
# ------------------------------------------------------------

DEBUG = True


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
# 4. 计算计数线长度
# ============================================================

line_x1, line_y1 = P1
line_x2, line_y2 = P2

line_dx = line_x2 - line_x1
line_dy = line_y2 - line_y1

line_length = math.sqrt(
    line_dx ** 2 +
    line_dy ** 2
)

if line_length == 0:

    print("\n错误：P1 和 P2 不能是同一个点。")

    exit()


print("\n计数线：")
print(f"P1 = {P1}")
print(f"P2 = {P2}")
print(f"线段长度 = {line_length:.2f} px")

print(
    f"确认距离 = {CONFIRM_DISTANCE} px"
)


# ============================================================
# 5. 读取轨迹
# ============================================================

df = pd.read_csv(
    CSV_PATH
)

print("\n")
print(f"轨迹数据量：{len(df)}")
print(
    f"Global ID 数量："
    f"{df['global_id'].nunique()}"
)


# ============================================================
# 6. 过滤过短轨迹
# ============================================================

global_counts = df[
    "global_id"
].value_counts()

valid_ids = global_counts[
    global_counts >= MIN_TRACK_FRAMES
].index

df = df[
    df["global_id"].isin(
        valid_ids
    )
].copy()

print(
    f"过滤后 Global ID："
    f"{df['global_id'].nunique()}"
)


# ============================================================
# 7. 检查坐标
# ============================================================

if "cx_smooth" not in df.columns:

    print(
        "\n错误：不存在 cx_smooth"
    )

    exit()


if "cy_smooth" not in df.columns:

    print(
        "\n错误：不存在 cy_smooth"
    )

    exit()


# ============================================================
# 8. 按 Global ID 排序
# ============================================================

df = df.sort_values(
    [
        "global_id",
        "frame"
    ]
).reset_index(
    drop=True
)


# ============================================================
# 9. 点在线段哪一侧
# ============================================================

def get_side(x, y):

    value = (
        line_dx * (y - line_y1)
        -
        line_dy * (x - line_x1)
    )

    if value > 0:

        return 1

    elif value < 0:

        return -1

    else:

        return 0


# ============================================================
# 10. 点到无限直线的垂直距离
# ============================================================

def distance_to_line(x, y):

    numerator = abs(
        line_dy * x
        -
        line_dx * y
        +
        line_x2 * line_y1
        -
        line_y2 * line_x1
    )

    return (
        numerator /
        line_length
    )


# ============================================================
# 11. 点到线段距离
# ============================================================

def distance_to_segment(x, y):

    px = x - line_x1
    py = y - line_y1

    length_sq = (
        line_dx ** 2 +
        line_dy ** 2
    )

    if length_sq == 0:

        return math.sqrt(
            px ** 2 +
            py ** 2
        )

    t = (
        px * line_dx +
        py * line_dy
    ) / length_sq

    t = max(
        0,
        min(1, t)
    )

    nearest_x = (
        line_x1 +
        t * line_dx
    )

    nearest_y = (
        line_y1 +
        t * line_dy
    )

    return math.sqrt(
        (x - nearest_x) ** 2 +
        (y - nearest_y) ** 2
    )


# ============================================================
# 12. 构建 Global ID 轨迹
# ============================================================

tracks = {}

for gid, group in df.groupby(
    "global_id"
):

    group = group.sort_values(
        "frame"
    )

    tracks[int(gid)] = {

        "frame":
            group["frame"].values,

        "time":
            group["time"].values,

        "cx":
            group["cx_smooth"].values,

        "cy":
            group["cy_smooth"].values
    }


# ============================================================
# 13. 开始计数
# ============================================================

events = []

IN_count = 0
OUT_count = 0

print("\n")
print("=" * 70)
print("开始分析 Global ID 穿越任意方向计数线")
print("=" * 70)
print()


# ============================================================
# 14. 遍历每一个 Global ID
# ============================================================

for gid, track in tracks.items():

    frames = track["frame"]
    times = track["time"]

    xs = track["cx"]
    ys = track["cy"]

    if len(frames) < MIN_TRACK_FRAMES:

        continue


    # --------------------------------------------------------
    # 初始侧
    # --------------------------------------------------------

    initial_side = None

    initial_index = None

    for i in range(len(xs)):

        side = get_side(
            xs[i],
            ys[i]
        )

        if side != 0:

            initial_side = side

            initial_index = i

            break


    if initial_side is None:

        continue


    current_side = initial_side

    # --------------------------------------------------------
    # 是否已经穿线
    # --------------------------------------------------------

    crossed = False

    # --------------------------------------------------------
    # 穿线信息
    # --------------------------------------------------------

    cross_frame = None
    cross_time = None
    cross_x = None
    cross_y = None

    # --------------------------------------------------------
    # 是否已经计数
    # --------------------------------------------------------

    counted = False


    # ========================================================
    # 遍历轨迹
    # ========================================================

    for i in range(
        initial_index + 1,
        len(xs)
    ):

        x_prev = float(
            xs[i - 1]
        )

        y_prev = float(
            ys[i - 1]
        )

        x_curr = float(
            xs[i]
        )

        y_curr = float(
            ys[i]
        )

        frame_curr = int(
            frames[i]
        )

        time_curr = float(
            times[i]
        )


        # ----------------------------------------------------
        # 当前点在线哪一侧
        # ----------------------------------------------------

        side_prev = get_side(
            x_prev,
            y_prev
        )

        side_curr = get_side(
            x_curr,
            y_curr
        )


        # ----------------------------------------------------
        # 忽略恰好在线上的点
        # ----------------------------------------------------

        if side_curr == 0:

            continue


        # ====================================================
        # A侧 → B侧
        # ====================================================

        if (
            not crossed
            and side_prev == current_side
            and side_curr != current_side
        ):

            crossed = True

            cross_frame = frame_curr

            cross_time = time_curr

            cross_x = x_curr

            cross_y = y_curr


            if DEBUG:

                print(
                    f"检测到穿线："
                    f"GID={gid} "
                    f"方向候选=IN "
                    f"time={time_curr:.2f}s "
                    f"位置=({x_curr:.1f},"
                    f"{y_curr:.1f})"
                )


        # ====================================================
        # 穿线以后确认
        # ====================================================

        if crossed:

            distance = distance_to_line(
                x_curr,
                y_curr
            )

            movement = math.sqrt(

                (x_curr - cross_x) ** 2
                +
                (y_curr - cross_y) ** 2

            )


            # ------------------------------------------------
            # 确认距离
            # ------------------------------------------------

            if (
                distance >= CONFIRM_DISTANCE
                and
                movement >= MIN_MOVEMENT_DISTANCE
            ):

                # =================================================
                # 判断最终方向
                # =================================================

                if side_curr != current_side:

                    # ------------------------------------------------
                    # A → B
                    # ------------------------------------------------

                    IN_count += 1

                    counted = True

                    events.append({

                        "global_id":
                            gid,

                        "direction":
                            "IN",

                        "frame":
                            frame_curr,

                        "time":
                            time_curr,

                        "cx":
                            x_curr,

                        "cy":
                            y_curr,

                        "cross_frame":
                            cross_frame,

                        "cross_time":
                            cross_time,

                        "cross_x":
                            cross_x,

                        "cross_y":
                            cross_y,

                        "distance_to_line":
                            distance,

                        "movement_after_cross":
                            movement
                    })


                    print(
                        f"IN  Global ID={gid:4d} "
                        f"time={time_curr:6.2f}s "
                        f"distance={distance:6.1f}px"
                    )

                    break


        # ----------------------------------------------------
        # 更新侧别
        # ----------------------------------------------------

        if not crossed:

            current_side = side_curr


# ============================================================
# 15. 保存事件 CSV
# ============================================================

events_df = pd.DataFrame(
    events
)


if len(events_df) > 0:

    events_df = events_df.sort_values(
        "frame"
    ).reset_index(
        drop=True
    )

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

            "distance_to_line",

            "movement_after_cross"
        ]
    )


events_df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 16. 输出统计结果
# ============================================================

print("\n")
print("=" * 70)
print("V5 任意方向计数完成")
print("=" * 70)

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

    print(
        "没有检测到有效穿线事件"
    )


print("\n事件文件：")
print(
    OUTPUT_CSV
)


# ============================================================
# 17. 打开原视频
# ============================================================

cap = cv2.VideoCapture(
    VIDEO_PATH
)

if not cap.isOpened():

    print(
        "\n无法打开视频。"
    )

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
    f"分辨率："
    f"{width} × {height}"
)

print(
    f"FPS：{fps:.2f}"
)

print(
    f"总帧数：{total_frames}"
)


# ============================================================
# 18. 创建输出视频
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
# 19. 按帧索引轨迹
# ============================================================

tracks_by_frame = defaultdict(
    list
)

for _, row in df.iterrows():

    frame = int(
        row["frame"]
    )

    tracks_by_frame[
        frame
    ].append(row)


# ============================================================
# 20. 按帧索引事件
# ============================================================

event_by_frame = defaultdict(
    list
)

for _, event in events_df.iterrows():

    frame = int(
        event["frame"]
    )

    event_by_frame[
        frame
    ].append(event)


# ============================================================
# 21. 生成视频
# ============================================================

frame_index = 1


while True:

    ret, frame = cap.read()

    if not ret:

        break


    # ========================================================
    # 绘制计数线
    # ========================================================

    cv2.line(
        frame,

        P1,

        P2,

        (255, 255, 0),

        4
    )


    # ========================================================
    # 绘制计数线端点
    # ========================================================

    cv2.circle(
        frame,
        P1,
        8,
        (0, 255, 255),
        -1
    )

    cv2.circle(
        frame,
        P2,
        8,
        (0, 255, 255),
        -1
    )


    # ========================================================
    # 绘制文字
    # ========================================================

    cv2.putText(

        frame,

        "COUNTING LINE",

        (
            P1[0],
            P1[1] - 15
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (255, 255, 0),

        2
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


            # ------------------------------------------------
            # 根据 Global ID 生成颜色
            # ------------------------------------------------

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

                (
                    cx + 8,
                    cy - 8
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.55,

                color,

                2
            )


    # ========================================================
    # 显示事件
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
                f"{direction} "
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

        (300, 140),

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
    # 写入视频
    # ========================================================

    writer.write(
        frame
    )

    frame_index += 1


# ============================================================
# 22. 释放资源
# ============================================================

cap.release()

writer.release()


# ============================================================
# 23. 完成
# ============================================================

print("\n")
print("=" * 70)
print("V5 结果视频生成完成")
print("=" * 70)

print(
    f"视频："
    f"{OUTPUT_VIDEO}"
)

print(
    f"事件："
    f"{OUTPUT_CSV}"
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

print("=" * 70)