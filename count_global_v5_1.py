import os
import cv2
import math
import pandas as pd
from collections import defaultdict


print("=" * 70)
print("Global ID 任意方向计数线统计 V5.1")
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
    "count_global_v5_1"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "global_count_events_v5_1.csv"
)

OUTPUT_VIDEO = os.path.join(
    OUTPUT_DIR,
    "global_count_result_v5_1.mp4"
)


# ============================================================
# 2. 计数线
# ============================================================

P1 = (200, 360)
P2 = (1100, 360)


# ============================================================
# 3. 参数
# ============================================================

MIN_TRACK_FRAMES = 10

# 穿线后需要离开计数线多少像素
CONFIRM_DISTANCE = 30

# 穿线后最少移动距离
MIN_MOVEMENT_DISTANCE = 20

# 计数线两端的有效范围
SEGMENT_MARGIN = 30


# ============================================================
# 4. 检查文件
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
# 5. 计数线参数
# ============================================================

x1, y1 = P1
x2, y2 = P2

dx = x2 - x1
dy = y2 - y1

line_length = math.sqrt(
    dx ** 2 +
    dy ** 2
)

if line_length == 0:

    print("错误：P1 和 P2 不能相同")

    exit()


print("\n计数线：")
print(f"P1 = {P1}")
print(f"P2 = {P2}")

print(
    f"计数线长度："
    f"{line_length:.2f}px"
)

print(
    f"确认距离："
    f"{CONFIRM_DISTANCE}px"
)


# ============================================================
# 6. 读取轨迹
# ============================================================

df = pd.read_csv(
    CSV_PATH
)

print("\n轨迹数据量：", len(df))

print(
    "Global ID 数量：",
    df["global_id"].nunique()
)


# ============================================================
# 7. 过滤短轨迹
# ============================================================

id_counts = df[
    "global_id"
].value_counts()

valid_ids = id_counts[
    id_counts >= MIN_TRACK_FRAMES
].index

df = df[
    df["global_id"].isin(
        valid_ids
    )
].copy()

print(
    "过滤后 Global ID：",
    df["global_id"].nunique()
)


# ============================================================
# 8. 使用平滑轨迹
# ============================================================

if (
    "cx_smooth" not in df.columns
    or
    "cy_smooth" not in df.columns
):

    print(
        "\n错误：轨迹文件中不存在平滑坐标"
    )

    exit()


df = df.sort_values(
    [
        "global_id",
        "frame"
    ]
)


# ============================================================
# 9. 点在线的哪一侧
# ============================================================

def get_side(x, y):

    value = (
        dx * (y - y1)
        -
        dy * (x - x1)
    )

    if value > 0:

        return 1

    elif value < 0:

        return -1

    return 0


# ============================================================
# 10. 点到直线距离
# ============================================================

def distance_to_line(x, y):

    numerator = abs(

        dy * x
        -
        dx * y
        +
        x2 * y1
        -
        y2 * x1

    )

    return (
        numerator /
        line_length
    )


# ============================================================
# 11. 判断投影点是否在线段范围内
# ============================================================

def is_on_segment(x, y):

    px = x - x1
    py = y - y1

    projection = (
        px * dx +
        py * dy
    ) / (
        line_length ** 2
    )

    return (
        -SEGMENT_MARGIN / line_length
        <= projection
        <=
        1 + SEGMENT_MARGIN / line_length
    )


# ============================================================
# 12. 建立 Global ID 轨迹
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

        "x":
            group["cx_smooth"].values,

        "y":
            group["cy_smooth"].values
    }


# ============================================================
# 13. 计数
# ============================================================

events = []

IN_count = 0
OUT_count = 0


print("\n")
print("=" * 70)
print("开始检测穿线")
print("=" * 70)
print()


# ============================================================
# 14. 遍历 Global ID
# ============================================================

for gid, track in tracks.items():

    frames = track["frame"]
    times = track["time"]

    xs = track["x"]
    ys = track["y"]

    if len(frames) < MIN_TRACK_FRAMES:

        continue


    # --------------------------------------------------------
    # 找到第一个有效侧
    # --------------------------------------------------------

    current_side = None

    start_index = None

    for i in range(len(xs)):

        side = get_side(
            xs[i],
            ys[i]
        )

        if side != 0:

            current_side = side

            start_index = i

            break


    if current_side is None:

        continue


    # --------------------------------------------------------
    # 防止一个 Global ID 重复计数
    # --------------------------------------------------------

    counted = False


    # ========================================================
    # 遍历轨迹
    # ========================================================

    for i in range(
        start_index + 1,
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
        # 当前点
        # ----------------------------------------------------

        side_prev = get_side(
            x_prev,
            y_prev
        )

        side_curr = get_side(
            x_curr,
            y_curr
        )


        if side_prev == 0:

            continue

        if side_curr == 0:

            continue


        # ====================================================
        # 判断是否发生穿线
        # ====================================================

        if (
            not counted
            and
            side_prev != side_curr
            and
            is_on_segment(
                x_curr,
                y_curr
            )
        ):

            cross_x = x_curr
            cross_y = y_curr

            cross_frame = frame_curr
            cross_time = time_curr

            old_side = current_side
            new_side = side_curr


            # ------------------------------------------------
            # 根据方向判断
            #
            # 当前水平线：
            #
            # 上 → 下 = IN
            # 下 → 上 = OUT
            # ------------------------------------------------

            # 上方 side = -1
            # 下方 side = +1

            if (
                old_side == -1
                and
                new_side == 1
            ):

                direction = "IN"

            elif (
                old_side == 1
                and
                new_side == -1
            ):

                direction = "OUT"

            else:

                direction = None


            if direction is None:

                current_side = side_curr

                continue


            print(
                f"检测到穿线："
                f"GID={gid} "
                f"候选={direction} "
                f"time={cross_time:.2f}s "
                f"位置=({cross_x:.1f},"
                f"{cross_y:.1f})"
            )


            # =================================================
            # 穿线后的确认
            # =================================================

            for j in range(
                i,
                len(xs)
            ):

                x_after = float(
                    xs[j]
                )

                y_after = float(
                    ys[j]
                )

                frame_after = int(
                    frames[j]
                )

                time_after = float(
                    times[j]
                )


                # ---------------------------------------------
                # 距离计数线
                # ---------------------------------------------

                distance = distance_to_line(
                    x_after,
                    y_after
                )


                # ---------------------------------------------
                # 穿线后的运动距离
                # ---------------------------------------------

                movement = math.sqrt(

                    (
                        x_after -
                        cross_x
                    ) ** 2

                    +

                    (
                        y_after -
                        cross_y
                    ) ** 2

                )


                # ---------------------------------------------
                # 侧别必须仍然保持在新侧
                # ---------------------------------------------

                side_after = get_side(
                    x_after,
                    y_after
                )


                if side_after != new_side:

                    continue


                # ---------------------------------------------
                # 达到确认条件
                # ---------------------------------------------

                if (
                    distance >=
                    CONFIRM_DISTANCE
                    and
                    movement >=
                    MIN_MOVEMENT_DISTANCE
                ):

                    counted = True


                    event = {

                        "global_id":
                            gid,

                        "direction":
                            direction,

                        "cross_frame":
                            cross_frame,

                        "cross_time":
                            cross_time,

                        "cross_x":
                            cross_x,

                        "cross_y":
                            cross_y,

                        "confirm_frame":
                            frame_after,

                        "confirm_time":
                            time_after,

                        "confirm_x":
                            x_after,

                        "confirm_y":
                            y_after,

                        "distance_to_line":
                            distance,

                        "movement_after_cross":
                            movement
                    }


                    events.append(
                        event
                    )


                    if direction == "IN":

                        IN_count += 1

                    else:

                        OUT_count += 1


                    print(
                        f"{direction:3s} "
                        f"Global ID={gid:4d} "
                        f"time={time_after:6.2f}s "
                        f"distance="
                        f"{distance:6.1f}px"
                    )


                    break


            if counted:

                break


        # ----------------------------------------------------
        # 更新侧
        # ----------------------------------------------------

        current_side = side_curr


# ============================================================
# 15. 保存 CSV
# ============================================================

events_df = pd.DataFrame(
    events
)

if len(events_df) > 0:

    events_df = events_df.sort_values(
        "confirm_frame"
    )

events_df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 16. 输出统计
# ============================================================

print("\n")
print("=" * 70)
print("V5.1 计数完成")
print("=" * 70)

print(
    f"IN  人数：{IN_count}"
)

print(
    f"OUT 人数：{OUT_count}"
)

print(
    f"TOTAL："
    f"{IN_count + OUT_count}"
)

print(
    f"\n事件文件："
    f"{OUTPUT_CSV}"
)


# ============================================================
# 17. 打开视频
# ============================================================

cap = cv2.VideoCapture(
    VIDEO_PATH
)

if not cap.isOpened():

    print("视频打开失败")

    exit()


fps = cap.get(
    cv2.CAP_PROP_FPS
)

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


fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (
        width,
        height
    )
)


# ============================================================
# 18. 轨迹按照 frame 建立索引
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
# 19. 事件按照 frame 建立索引
# ============================================================

events_by_frame = defaultdict(
    list
)

for event in events:

    frame = int(
        event["confirm_frame"]
    )

    events_by_frame[
        frame
    ].append(
        event
    )


# ============================================================
# 20. 生成结果视频
# ============================================================

frame_index = 1


while True:

    ret, frame = cap.read()

    if not ret:

        break


    # ========================================================
    # 计数线
    # ========================================================

    cv2.line(
        frame,
        P1,
        P2,
        (255, 255, 0),
        4
    )


    # ========================================================
    # 端点
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
    # 方向文字
    # ========================================================

    cv2.putText(

        frame,

        "IN: TOP -> BOTTOM",

        (
            20,
            height - 60
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (0, 255, 0),

        2
    )


    cv2.putText(

        frame,

        "OUT: BOTTOM -> TOP",

        (
            20,
            height - 25
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (0, 165, 255),

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


            color = (

                (gid * 37) % 255,

                (gid * 67) % 255,

                (gid * 97) % 255
            )


            cv2.circle(

                frame,

                (
                    cx,
                    cy
                ),

                5,

                color,

                -1
            )


            cv2.putText(

                frame,

                f"GID:{gid}",

                (
                    cx + 7,
                    cy - 7
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.5,

                color,

                2
            )


    # ========================================================
    # 显示事件
    # ========================================================

    if frame_index in events_by_frame:

        for event in events_by_frame[
            frame_index
        ]:

            direction = event[
                "direction"
            ]

            gid = int(
                event["global_id"]
            )


            cv2.putText(

                frame,

                f"{direction} "
                f"GID:{gid}",

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
    # 统计面板
    # ========================================================

    cv2.rectangle(

        frame,

        (
            10,
            10
        ),

        (
            310,
            145
        ),

        (0, 0, 0),

        -1
    )


    cv2.putText(

        frame,

        f"IN: {IN_count}",

        (
            25,
            45
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        (0, 255, 0),

        2
    )


    cv2.putText(

        frame,

        f"OUT: {OUT_count}",

        (
            25,
            80
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        (0, 165, 255),

        2
    )


    cv2.putText(

        frame,

        f"TOTAL: "
        f"{IN_count + OUT_count}",

        (
            25,
            115
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        (255, 255, 255),

        2
    )


    # ========================================================
    # 写入
    # ========================================================

    writer.write(
        frame
    )

    frame_index += 1


# ============================================================
# 21. 释放
# ============================================================

cap.release()

writer.release()


print("\n")
print("=" * 70)
print("V5.1 结果视频生成完成")
print("=" * 70)

print(
    "视频："
    f"{OUTPUT_VIDEO}"
)

print(
    "CSV："
    f"{OUTPUT_CSV}"
)

print(
    "\n最终结果："
)

print(
    f"IN = {IN_count}"
)

print(
    f"OUT = {OUT_count}"
)

print(
    f"TOTAL = "
    f"{IN_count + OUT_count}"
)

print("=" * 70)