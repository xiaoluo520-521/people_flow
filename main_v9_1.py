import os
import json
import math
import cv2
import numpy as np
import pandas as pd
from collections import defaultdict


# ============================================================
# 1. 路径配置
# ============================================================

BASE_DIR = r"E:\people_flow"

CONFIG_FILE = os.path.join(
    BASE_DIR,
    "config.json"
)

TRAJECTORY_FILE = os.path.join(
    BASE_DIR,
    "results",
    "trajectory_reconnect_v2",
    "reconnected_tracks.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "unified_v9_1"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. 输出文件
# ============================================================

COUNT_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "count_events.csv"
)

DWELL_EVENT_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "dwell_events.csv"
)

DWELL_SUMMARY_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "dwell_summary.csv"
)

HEATMAP_DATA_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "heatmap_data.csv"
)

HEATMAP_IMAGE_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "heatmap_final.png"
)

SUMMARY_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "analysis_summary.json"
)

VIDEO_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "analysis_result.mp4"
)


# ============================================================
# 3. 读取配置
# ============================================================

print("=" * 70)
print("V9.1 统一人流分析引擎")
print("=" * 70)

if not os.path.exists(CONFIG_FILE):
    raise FileNotFoundError(
        f"找不到配置文件：{CONFIG_FILE}\n"
        f"请先运行 config_v9.py"
    )

with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    config = json.load(f)

video_path = config["video"]

line_p1 = tuple(
    map(float, config["counting_line"]["p1"])
)

line_p2 = tuple(
    map(float, config["counting_line"]["p2"])
)

roi_points = [
    tuple(map(int, p))
    for p in config["roi"]
]

params = config.get("parameters", {})

MIN_TRACK_FRAMES = int(
    params.get("min_track_frames", 10)
)

CONFIRM_DISTANCE = float(
    params.get("confirm_distance", 30)
)

MIN_MOVEMENT_DISTANCE = float(
    params.get("min_movement_distance", 20)
)

MAX_FRAME_GAP = int(
    params.get("max_frame_gap", 2)
)

HEATMAP_BLUR_KERNEL = int(
    params.get("heatmap_blur_kernel", 51)
)

# 保证高斯核为奇数
if HEATMAP_BLUR_KERNEL % 2 == 0:
    HEATMAP_BLUR_KERNEL += 1


print("\n配置读取成功")
print(f"视频：{video_path}")
print(f"计数线：{line_p1} -> {line_p2}")
print(f"ROI点数：{len(roi_points)}")
print(f"最小轨迹长度：{MIN_TRACK_FRAMES}")
print(f"确认距离：{CONFIRM_DISTANCE}")
print(f"最小运动距离：{MIN_MOVEMENT_DISTANCE}")
print(f"最大允许帧间隔：{MAX_FRAME_GAP}")


# ============================================================
# 4. 检查轨迹文件
# ============================================================

if not os.path.exists(TRAJECTORY_FILE):
    raise FileNotFoundError(
        f"找不到轨迹文件：{TRAJECTORY_FILE}"
    )

df = pd.read_csv(TRAJECTORY_FILE)

required_columns = [
    "frame",
    "time",
    "global_id",
    "cx_smooth",
    "cy_smooth"
]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(
            f"轨迹文件缺少字段：{col}"
        )

df = df.sort_values(
    ["global_id", "frame"]
).reset_index(drop=True)

print("\n轨迹数据")
print(f"数据量：{len(df)}")
print(
    f"Global ID："
    f"{df['global_id'].nunique()}"
)


# ============================================================
# 5. 过滤短轨迹
# ============================================================

track_counts = df.groupby(
    "global_id"
).size()

valid_ids = track_counts[
    track_counts >= MIN_TRACK_FRAMES
].index

df_valid = df[
    df["global_id"].isin(valid_ids)
].copy()

print(
    f"过滤后有效 Global ID："
    f"{df_valid['global_id'].nunique()}"
)


# ============================================================
# 6. 任意方向计数线数学函数
# ============================================================

P1 = np.array(line_p1, dtype=np.float64)
P2 = np.array(line_p2, dtype=np.float64)

LINE_VECTOR = P2 - P1

LINE_LENGTH = np.linalg.norm(
    LINE_VECTOR
)

if LINE_LENGTH < 1:
    raise ValueError("计数线长度不能为0")


def point_side(point):
    """
    判断点在线的哪一侧。

    > 0：正侧
    < 0：负侧
    """

    p = np.array(point, dtype=np.float64)

    return (
        LINE_VECTOR[0] * (p[1] - P1[1])
        -
        LINE_VECTOR[1] * (p[0] - P1[0])
    )


def distance_to_line(point):
    """
    点到无限延长线的距离。
    """

    return abs(
        point_side(point)
    ) / LINE_LENGTH


def projection_ratio(point):
    """
    判断点在线段 P1-P2 上的投影位置。

    0   -> P1
    1   -> P2
    """

    p = np.array(point, dtype=np.float64)

    return np.dot(
        p - P1,
        LINE_VECTOR
    ) / (
        LINE_LENGTH ** 2
    )


def near_line_segment(point, margin=50):
    """
    判断穿线点是否位于计数线附近。
    """

    ratio = projection_ratio(point)

    return (
        -margin / LINE_LENGTH
        <= ratio
        <=
        1 + margin / LINE_LENGTH
    )


# ============================================================
# 7. 进出计数
# ============================================================

print("\n开始进行进出计数...")

count_events = []

grouped_tracks = df_valid.groupby(
    "global_id"
)

for global_id, group in grouped_tracks:

    group = group.sort_values(
        "frame"
    ).reset_index(drop=True)

    if len(group) < MIN_TRACK_FRAMES:
        continue

    points = group[
        ["cx_smooth", "cy_smooth"]
    ].values

    frames = group[
        "frame"
    ].values

    times = group[
        "time"
    ].values

    sides = [
        point_side(p)
        for p in points
    ]

    counted = False

    for i in range(1, len(points)):

        if counted:
            break

        p_prev = points[i - 1]
        p_curr = points[i]

        frame_prev = int(frames[i - 1])
        frame_curr = int(frames[i])

        # ----------------------------------------------------
        # 帧间不能有过大的断裂
        # ----------------------------------------------------

        if frame_curr - frame_prev > MAX_FRAME_GAP:
            continue

        # ----------------------------------------------------
        # 位移距离
        # ----------------------------------------------------

        movement = np.linalg.norm(
            p_curr - p_prev
        )

        if movement < MIN_MOVEMENT_DISTANCE:
            continue

        side_prev = sides[i - 1]
        side_curr = sides[i]

        # ----------------------------------------------------
        # 必须真正穿过直线
        # ----------------------------------------------------

        if side_prev == 0 or side_curr == 0:
            continue

        if side_prev * side_curr >= 0:
            continue

        # ----------------------------------------------------
        # 计算穿线点
        # ----------------------------------------------------

        denom = abs(side_prev) + abs(side_curr)

        if denom == 0:
            continue

        ratio = abs(side_prev) / denom

        cross_point = (
            p_prev
            +
            ratio * (p_curr - p_prev)
        )

        # ----------------------------------------------------
        # 穿线位置必须靠近真正的线段
        # ----------------------------------------------------

        if not near_line_segment(
            cross_point,
            margin=50
        ):
            continue

        cross_frame = (
            frame_prev
            +
            ratio *
            (frame_curr - frame_prev)
        )

        cross_time = (
            times[i - 1]
            +
            ratio *
            (times[i] - times[i - 1])
        )

        # ----------------------------------------------------
        # 判断方向
        #
        # 默认：
        #
        # 负侧 -> 正侧 = IN
        # 正侧 -> 负侧 = OUT
        # ----------------------------------------------------

        if side_prev < 0 and side_curr > 0:
            direction = "IN"

        elif side_prev > 0 and side_curr < 0:
            direction = "OUT"

        else:
            continue

        # ----------------------------------------------------
        # 后续确认
        #
        # 必须离开计数线一定距离，
        # 才真正确定这不是抖动。
        # ----------------------------------------------------

        confirmed = False

        confirm_index = None

        for j in range(i, len(points)):

            if frames[j] - frames[i] > 30:
                break

            current_distance = distance_to_line(
                points[j]
            )

            if current_distance >= CONFIRM_DISTANCE:

                # 必须仍然位于目标侧
                target_side = point_side(
                    points[j]
                )

                if direction == "IN" and target_side > 0:
                    confirmed = True
                    confirm_index = j
                    break

                if direction == "OUT" and target_side < 0:
                    confirmed = True
                    confirm_index = j
                    break

        if not confirmed:
            continue

        confirm_point = points[
            confirm_index
        ]

        confirm_frame = int(
            frames[confirm_index]
        )

        confirm_time = float(
            times[confirm_index]
        )

        cross_distance = distance_to_line(
            confirm_point
        )

        count_events.append({
            "global_id": int(global_id),
            "direction": direction,

            "cross_frame": float(cross_frame),
            "cross_time": float(cross_time),
            "cross_x": float(cross_point[0]),
            "cross_y": float(cross_point[1]),

            "confirm_frame": confirm_frame,
            "confirm_time": confirm_time,
            "confirm_x": float(confirm_point[0]),
            "confirm_y": float(confirm_point[1]),

            "confirm_distance": float(
                cross_distance
            )
        })

        counted = True


# ============================================================
# 8. 保存计数结果
# ============================================================

count_df = pd.DataFrame(
    count_events
)

if len(count_df) > 0:

    count_df = count_df.sort_values(
        "cross_time"
    ).reset_index(drop=True)

    count_df.to_csv(
        COUNT_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

else:

    count_df = pd.DataFrame(
        columns=[
            "global_id",
            "direction",
            "cross_frame",
            "cross_time",
            "cross_x",
            "cross_y",
            "confirm_frame",
            "confirm_time",
            "confirm_x",
            "confirm_y",
            "confirm_distance"
        ]
    )

    count_df.to_csv(
        COUNT_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )


IN_COUNT = int(
    (count_df["direction"] == "IN").sum()
)

OUT_COUNT = int(
    (count_df["direction"] == "OUT").sum()
)

TOTAL_COUNT = (
    IN_COUNT
    +
    OUT_COUNT
)

print("\n进出计数完成")
print(f"IN：{IN_COUNT}")
print(f"OUT：{OUT_COUNT}")
print(f"TOTAL：{TOTAL_COUNT}")


# ============================================================
# 9. ROI 区域停留分析
# ============================================================

print("\n开始区域停留时间分析...")

roi_np = np.array(
    roi_points,
    dtype=np.int32
)


def point_in_roi(x, y):

    result = cv2.pointPolygonTest(
        roi_np,
        (float(x), float(y)),
        False
    )

    return result >= 0


dwell_events = []


for global_id, group in grouped_tracks:

    group = group.sort_values(
        "frame"
    ).reset_index(drop=True)

    inside = False

    enter_frame = None
    enter_time = None

    last_inside_frame = None
    last_inside_time = None

    visit_index = 0

    for i, row in group.iterrows():

        frame = int(row["frame"])
        time = float(row["time"])

        x = float(row["cx_smooth"])
        y = float(row["cy_smooth"])

        current_inside = point_in_roi(
            x,
            y
        )

        # ----------------------------------------------------
        # 判断轨迹是否发生断裂
        # ----------------------------------------------------

        if i > 0:

            prev_frame = int(
                group.loc[i - 1, "frame"]
            )

            frame_gap = (
                frame - prev_frame
            )

            if frame_gap > MAX_FRAME_GAP:

                # 如果此前在 ROI 中，
                # 先关闭此次访问
                if inside:

                    dwell_time = (
                        last_inside_time
                        -
                        enter_time
                    )

                    visit_index += 1

                    dwell_events.append({
                        "global_id": int(global_id),
                        "visit_id": visit_index,
                        "enter_frame": enter_frame,
                        "enter_time": enter_time,
                        "exit_frame": last_inside_frame,
                        "exit_time": last_inside_time,
                        "dwell_time": max(
                            0,
                            dwell_time
                        )
                    })

                    inside = False

                    enter_frame = None
                    enter_time = None

        # ----------------------------------------------------
        # ROI 外 -> ROI 内
        # ----------------------------------------------------

        if not inside and current_inside:

            inside = True

            enter_frame = frame
            enter_time = time

            last_inside_frame = frame
            last_inside_time = time

        # ----------------------------------------------------
        # ROI 内 -> ROI 外
        # ----------------------------------------------------

        elif inside and not current_inside:

            dwell_time = (
                last_inside_time
                -
                enter_time
            )

            visit_index += 1

            dwell_events.append({
                "global_id": int(global_id),
                "visit_id": visit_index,
                "enter_frame": enter_frame,
                "enter_time": enter_time,
                "exit_frame": frame,
                "exit_time": time,
                "dwell_time": max(
                    0,
                    dwell_time
                )
            })

            inside = False

            enter_frame = None
            enter_time = None

        # ----------------------------------------------------
        # ROI 内持续停留
        # ----------------------------------------------------

        if inside and current_inside:

            last_inside_frame = frame
            last_inside_time = time

    # --------------------------------------------------------
    # 视频结束时仍在 ROI 内
    # --------------------------------------------------------

    if inside:

        dwell_time = (
            last_inside_time
            -
            enter_time
        )

        visit_index += 1

        dwell_events.append({
            "global_id": int(global_id),
            "visit_id": visit_index,
            "enter_frame": enter_frame,
            "enter_time": enter_time,
            "exit_frame": last_inside_frame,
            "exit_time": last_inside_time,
            "dwell_time": max(
                0,
                dwell_time
            )
        })


dwell_df = pd.DataFrame(
    dwell_events
)


if len(dwell_df) > 0:

    dwell_df = dwell_df.sort_values(
        ["global_id", "enter_time"]
    ).reset_index(drop=True)

    dwell_df.to_csv(
        DWELL_EVENT_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

else:

    dwell_df = pd.DataFrame(
        columns=[
            "global_id",
            "visit_id",
            "enter_frame",
            "enter_time",
            "exit_frame",
            "exit_time",
            "dwell_time"
        ]
    )

    dwell_df.to_csv(
        DWELL_EVENT_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# 10. ROI 汇总
# ============================================================

if len(dwell_df) > 0:

    dwell_summary = (
        dwell_df
        .groupby("global_id")
        .agg(
            visit_count=(
                "visit_id",
                "count"
            ),
            total_dwell_time=(
                "dwell_time",
                "sum"
            ),
            avg_dwell_time=(
                "dwell_time",
                "mean"
            ),
            max_dwell_time=(
                "dwell_time",
                "max"
            )
        )
        .reset_index()
    )

else:

    dwell_summary = pd.DataFrame(
        columns=[
            "global_id",
            "visit_count",
            "total_dwell_time",
            "avg_dwell_time",
            "max_dwell_time"
        ]
    )


dwell_summary.to_csv(
    DWELL_SUMMARY_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


REGION_VISITORS = int(
    dwell_summary["global_id"].nunique()
)

REGION_VISITS = int(
    len(dwell_df)
)

if len(dwell_df) > 0:

    AVG_DWELL = float(
        dwell_df["dwell_time"].mean()
    )

    MAX_DWELL = float(
        dwell_df["dwell_time"].max()
    )

    TOTAL_DWELL = float(
        dwell_df["dwell_time"].sum()
    )

else:

    AVG_DWELL = 0
    MAX_DWELL = 0
    TOTAL_DWELL = 0


print("\n区域分析完成")
print(f"区域访问人数：{REGION_VISITORS}")
print(f"区域访问次数：{REGION_VISITS}")
print(f"平均停留时间：{AVG_DWELL:.2f}s")
print(f"最长停留时间：{MAX_DWELL:.2f}s")
print(f"总停留时间：{TOTAL_DWELL:.2f}s")


# ============================================================
# 11. 热力图数据
# ============================================================

print("\n开始生成热力图...")

heatmap_points = df_valid[
    [
        "global_id",
        "frame",
        "time",
        "cx_smooth",
        "cy_smooth"
    ]
].copy()

heatmap_points.to_csv(
    HEATMAP_DATA_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 12. 获取视频参数
# ============================================================

if not os.path.exists(video_path):
    raise FileNotFoundError(
        f"找不到视频：{video_path}"
    )

cap = cv2.VideoCapture(
    video_path
)

if not cap.isOpened():
    raise RuntimeError(
        "视频打开失败"
    )

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

fps = cap.get(
    cv2.CAP_PROP_FPS
)

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

if fps <= 0:
    fps = 30


print("\n视频信息")
print(f"分辨率：{width} × {height}")
print(f"FPS：{fps:.2f}")
print(f"总帧数：{total_frames}")


# ============================================================
# 13. 生成静态热力图
# ============================================================

heatmap = np.zeros(
    (height, width),
    dtype=np.float32
)

for _, row in heatmap_points.iterrows():

    x = int(
        round(row["cx_smooth"])
    )

    y = int(
        round(row["cy_smooth"])
    )

    if (
        0 <= x < width
        and
        0 <= y < height
    ):

        heatmap[y, x] += 1


# 平滑
heatmap_blur = cv2.GaussianBlur(
    heatmap,
    (
        HEATMAP_BLUR_KERNEL,
        HEATMAP_BLUR_KERNEL
    ),
    0
)


# 归一化
if heatmap_blur.max() > 0:

    heatmap_norm = cv2.normalize(
        heatmap_blur,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

else:

    heatmap_norm = np.zeros(
        (height, width),
        dtype=np.uint8
    )


heatmap_color = cv2.applyColorMap(
    heatmap_norm,
    cv2.COLORMAP_JET
)


# 原始画面
cap.set(
    cv2.CAP_PROP_POS_FRAMES,
    0
)

ret, first_frame = cap.read()

if not ret:
    raise RuntimeError(
        "无法读取视频第一帧"
    )


heatmap_static = cv2.addWeighted(
    first_frame,
    0.45,
    heatmap_color,
    0.55,
    0
)

cv2.polylines(
    heatmap_static,
    [roi_np],
    True,
    (255, 255, 255),
    3
)

cv2.line(
    heatmap_static,
    tuple(map(int, P1)),
    tuple(map(int, P2)),
    (255, 255, 255),
    3
)

cv2.imwrite(
    HEATMAP_IMAGE_OUTPUT,
    heatmap_static
)


print(
    f"静态热力图：{HEATMAP_IMAGE_OUTPUT}"
)


# ============================================================
# 14. 为视频准备轨迹索引
# ============================================================

frame_tracks = defaultdict(list)

for _, row in df_valid.iterrows():

    frame_id = int(
        row["frame"]
    )

    frame_tracks[frame_id].append({
        "global_id": int(
            row["global_id"]
        ),
        "x": float(
            row["cx_smooth"]
        ),
        "y": float(
            row["cy_smooth"]
        )
    })


# ============================================================
# 15. ROI 当前人数
# ============================================================

def get_current_roi_people(
    frame_id
):

    people = frame_tracks.get(
        frame_id,
        []
    )

    result = []

    for p in people:

        if point_in_roi(
            p["x"],
            p["y"]
        ):
            result.append(p)

    return result


# ============================================================
# 16. 视频输出
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    VIDEO_OUTPUT,
    fourcc,
    fps,
    (width, height)
)


# ============================================================
# 17. 历史轨迹
# ============================================================

trail_history = defaultdict(list)

TRAIL_LENGTH = 30


# ============================================================
# 18. 主视频循环
# ============================================================

print("\n开始生成统一分析视频...")

cap.set(
    cv2.CAP_PROP_POS_FRAMES,
    0
)

frame_number = 0


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # --------------------------------------------------------
    # 当前人
    # --------------------------------------------------------

    current_people = frame_tracks.get(
        frame_number,
        []
    )

    # --------------------------------------------------------
    # 更新轨迹
    # --------------------------------------------------------

    for p in current_people:

        gid = p["global_id"]

        point = (
            int(p["x"]),
            int(p["y"])
        )

        trail_history[gid].append(
            point
        )

        if len(
            trail_history[gid]
        ) > TRAIL_LENGTH:

            trail_history[gid] = \
                trail_history[gid][
                    -TRAIL_LENGTH:
                ]

    # --------------------------------------------------------
    # 热力图背景
    # --------------------------------------------------------

    display = cv2.addWeighted(
        frame,
        0.65,
        heatmap_color,
        0.35,
        0
    )

    # --------------------------------------------------------
    # 绘制 ROI
    # --------------------------------------------------------

    cv2.polylines(
        display,
        [roi_np],
        True,
        (255, 255, 255),
        3
    )

    # ROI 标签
    roi_center = np.mean(
        roi_np,
        axis=0
    ).astype(int)

    cv2.putText(
        display,
        "ROI",
        tuple(roi_center),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # 绘制计数线
    # --------------------------------------------------------

    cv2.line(
        display,
        tuple(map(int, P1)),
        tuple(map(int, P2)),
        (255, 255, 255),
        4
    )

    # P1 / P2
    cv2.circle(
        display,
        tuple(map(int, P1)),
        7,
        (255, 255, 255),
        -1
    )

    cv2.circle(
        display,
        tuple(map(int, P2)),
        7,
        (255, 255, 255),
        -1
    )

    cv2.putText(
        display,
        "COUNTING LINE",
        tuple(map(int, P1)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # 绘制轨迹
    # --------------------------------------------------------

    for gid, points in trail_history.items():

        if len(points) < 2:
            continue

        for k in range(
            1,
            len(points)
        ):

            cv2.line(
                display,
                points[k - 1],
                points[k],
                (0, 255, 255),
                2
            )

    # --------------------------------------------------------
    # 当前目标
    # --------------------------------------------------------

    roi_people = 0

    for p in current_people:

        gid = p["global_id"]

        x = int(p["x"])
        y = int(p["y"])

        in_roi = point_in_roi(
            x,
            y
        )

        if in_roi:
            roi_people += 1

        # 当前点
        cv2.circle(
            display,
            (x, y),
            5,
            (0, 255, 0)
            if in_roi
            else (255, 255, 0),
            -1
        )

        # ID
        cv2.putText(
            display,
            f"GID {gid}",
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )

    # --------------------------------------------------------
    # 当前时间
    # --------------------------------------------------------

    current_time = (
        frame_number / fps
    )

    # --------------------------------------------------------
    # 信息面板
    # --------------------------------------------------------

    overlay = display.copy()

    cv2.rectangle(
        overlay,
        (15, 15),
        (390, 175),
        (0, 0, 0),
        -1
    )

    display = cv2.addWeighted(
        overlay,
        0.65,
        display,
        0.35,
        0
    )

    cv2.putText(
        display,
        f"Time: {current_time:.2f}s",
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display,
        f"IN: {IN_COUNT}",
        (30, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display,
        f"OUT: {OUT_COUNT}",
        (150, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display,
        f"TOTAL: {TOTAL_COUNT}",
        (270, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display,
        f"ROI People: {roi_people}",
        (30, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display,
        f"Region Visitors: {REGION_VISITORS}",
        (30, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display,
        f"Avg Dwell: {AVG_DWELL:.1f}s",
        (30, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    writer.write(
        display
    )

    if frame_number % 100 == 0:

        print(
            f"视频处理："
            f"{frame_number}/{total_frames}"
        )


# ============================================================
# 19. 释放视频
# ============================================================

cap.release()
writer.release()


# ============================================================
# 20. 保存最终统计
# ============================================================

summary = {

    "video": video_path,

    "video_info": {
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": total_frames
    },

    "trajectory": {
        "original_global_ids":
            int(df["global_id"].nunique()),

        "valid_global_ids":
            int(df_valid["global_id"].nunique()),

        "trajectory_points":
            int(len(df_valid))
    },

    "counting": {

        "IN":
            IN_COUNT,

        "OUT":
            OUT_COUNT,

        "TOTAL":
            TOTAL_COUNT
    },

    "region": {

        "visitors":
            REGION_VISITORS,

        "visits":
            REGION_VISITS,

        "average_dwell_time":
            AVG_DWELL,

        "maximum_dwell_time":
            MAX_DWELL,

        "total_dwell_time":
            TOTAL_DWELL
    },

    "configuration": {

        "counting_line": {
            "p1": list(line_p1),
            "p2": list(line_p2)
        },

        "roi": [
            list(p)
            for p in roi_points
        ],

        "parameters": params
    },

    "outputs": {

        "count_events":
            COUNT_OUTPUT,

        "dwell_events":
            DWELL_EVENT_OUTPUT,

        "dwell_summary":
            DWELL_SUMMARY_OUTPUT,

        "heatmap_data":
            HEATMAP_DATA_OUTPUT,

        "heatmap_image":
            HEATMAP_IMAGE_OUTPUT,

        "analysis_video":
            VIDEO_OUTPUT
    }
}


with open(
    SUMMARY_OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        ensure_ascii=False,
        indent=4
    )


# ============================================================
# 21. 最终输出
# ============================================================

print("\n")
print("=" * 70)
print("V9.1 统一分析完成")
print("=" * 70)

print(f"IN：{IN_COUNT}")
print(f"OUT：{OUT_COUNT}")
print(f"TOTAL：{TOTAL_COUNT}")

print(f"\n区域访问人数：{REGION_VISITORS}")
print(f"区域访问次数：{REGION_VISITS}")
print(f"平均停留：{AVG_DWELL:.2f}s")
print(f"最长停留：{MAX_DWELL:.2f}s")

print("\n输出文件：")
print(COUNT_OUTPUT)
print(DWELL_EVENT_OUTPUT)
print(DWELL_SUMMARY_OUTPUT)
print(HEATMAP_DATA_OUTPUT)
print(HEATMAP_IMAGE_OUTPUT)
print(SUMMARY_OUTPUT)
print(VIDEO_OUTPUT)

print("=" * 70)