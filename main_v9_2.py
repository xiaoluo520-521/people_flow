import os
import json
import cv2
import numpy as np
import pandas as pd
from collections import defaultdict


# ============================================================
# 1. 路径
# ============================================================

BASE_DIR = r"E:\people_flow"

CONFIG_FILE = os.path.join(
    BASE_DIR, "config.json"
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
    "unified_v9_2"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. 输出文件
# ============================================================

COUNT_OUTPUT = os.path.join(
    OUTPUT_DIR, "count_events_v9_2.csv"
)

COUNT_DEBUG_OUTPUT = os.path.join(
    OUTPUT_DIR, "count_debug_v9_2.csv"
)

DWELL_EVENT_OUTPUT = os.path.join(
    OUTPUT_DIR, "dwell_events_v9_2.csv"
)

DWELL_SUMMARY_OUTPUT = os.path.join(
    OUTPUT_DIR, "dwell_summary_v9_2.csv"
)

HEATMAP_DATA_OUTPUT = os.path.join(
    OUTPUT_DIR, "heatmap_data_v9_2.csv"
)

HEATMAP_IMAGE_OUTPUT = os.path.join(
    OUTPUT_DIR, "heatmap_final_v9_2.png"
)

SUMMARY_OUTPUT = os.path.join(
    OUTPUT_DIR, "analysis_summary_v9_2.json"
)

VIDEO_OUTPUT = os.path.join(
    OUTPUT_DIR, "analysis_result_v9_2.mp4"
)


# ============================================================
# 3. 读取配置
# ============================================================

print("=" * 72)
print("V9.2 稳健型人流统一分析引擎")
print("=" * 72)

if not os.path.exists(CONFIG_FILE):
    raise FileNotFoundError(
        f"找不到配置文件：{CONFIG_FILE}"
    )

with open(
    CONFIG_FILE,
    "r",
    encoding="utf-8"
) as f:
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

params = config.get(
    "parameters",
    {}
)


MIN_TRACK_FRAMES = int(
    params.get("min_track_frames", 10)
)

CONFIRM_DISTANCE = float(
    params.get("confirm_distance", 20)
)

MIN_MOVEMENT_DISTANCE = float(
    params.get("min_movement_distance", 10)
)

MAX_FRAME_GAP = int(
    params.get("max_frame_gap", 5)
)

HEATMAP_BLUR_KERNEL = int(
    params.get("heatmap_blur_kernel", 51)
)

if HEATMAP_BLUR_KERNEL % 2 == 0:
    HEATMAP_BLUR_KERNEL += 1


# ============================================================
# ROI 参数
# ============================================================

# ROI内部允许短暂轨迹丢失的时间
ROI_MAX_GAP = 10

# 小于这个时间的访问认为可能是噪声
MIN_DWELL_TIME = 0.5


print("\n配置读取成功")
print(f"视频：{video_path}")
print(
    f"计数线："
    f"{line_p1} -> {line_p2}"
)
print(f"ROI点数：{len(roi_points)}")

print("\n计数参数")
print(f"最小轨迹长度：{MIN_TRACK_FRAMES}")
print(f"确认距离：{CONFIRM_DISTANCE}px")
print(
    f"最小运动距离："
    f"{MIN_MOVEMENT_DISTANCE}px"
)
print(
    f"最大轨迹帧间隔："
    f"{MAX_FRAME_GAP}"
)

print("\nROI参数")
print(
    f"ROI允许短暂丢失："
    f"{ROI_MAX_GAP}帧"
)
print(
    f"最小有效停留时间："
    f"{MIN_DWELL_TIME}s"
)


# ============================================================
# 4. 读取轨迹
# ============================================================

if not os.path.exists(
    TRAJECTORY_FILE
):
    raise FileNotFoundError(
        f"找不到轨迹文件："
        f"{TRAJECTORY_FILE}"
    )

df = pd.read_csv(
    TRAJECTORY_FILE
)

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
            f"缺少字段：{col}"
        )


df = df.sort_values(
    ["global_id", "frame"]
).reset_index(drop=True)


print("\n轨迹数据")
print(
    f"轨迹点：{len(df)}"
)

print(
    f"Global ID："
    f"{df['global_id'].nunique()}"
)


# ============================================================
# 5. 过滤短轨迹
# ============================================================

track_counts = (
    df.groupby("global_id")
    .size()
)

valid_ids = track_counts[
    track_counts >= MIN_TRACK_FRAMES
].index

df_valid = df[
    df["global_id"].isin(valid_ids)
].copy()

print(
    f"有效 Global ID："
    f"{df_valid['global_id'].nunique()}"
)


# ============================================================
# 6. 计数线
# ============================================================

P1 = np.array(
    line_p1,
    dtype=np.float64
)

P2 = np.array(
    line_p2,
    dtype=np.float64
)

LINE_VECTOR = P2 - P1

LINE_LENGTH = np.linalg.norm(
    LINE_VECTOR
)

if LINE_LENGTH < 1:
    raise ValueError(
        "计数线长度不能为0"
    )


def point_side(point):

    p = np.array(
        point,
        dtype=np.float64
    )

    return (
        LINE_VECTOR[0]
        *
        (p[1] - P1[1])
        -
        LINE_VECTOR[1]
        *
        (p[0] - P1[0])
    )


def distance_to_line(point):

    return abs(
        point_side(point)
    ) / LINE_LENGTH


def projection_ratio(point):

    p = np.array(
        point,
        dtype=np.float64
    )

    return np.dot(
        p - P1,
        LINE_VECTOR
    ) / (
        LINE_LENGTH ** 2
    )


def near_line_segment(
    point,
    margin=100
):

    ratio = projection_ratio(
        point
    )

    return (
        ratio >= -margin / LINE_LENGTH
        and
        ratio <=
        1 + margin / LINE_LENGTH
    )


def interpolate_crossing(
    p1,
    p2,
    side1,
    side2
):

    denom = (
        abs(side1)
        +
        abs(side2)
    )

    if denom < 1e-8:
        return p1.copy(), 0.0

    ratio = (
        abs(side1)
        /
        denom
    )

    point = (
        p1
        +
        ratio *
        (p2 - p1)
    )

    return point, ratio


# ============================================================
# 7. 稳健穿线分析
# ============================================================

print("\n开始计数线分析...")


count_events = []

debug_records = []


for global_id, group in (
    df_valid.groupby("global_id")
):

    group = group.sort_values(
        "frame"
    ).reset_index(drop=True)

    if len(group) < MIN_TRACK_FRAMES:
        continue

    points = group[
        [
            "cx_smooth",
            "cy_smooth"
        ]
    ].values.astype(float)

    frames = group[
        "frame"
    ].values

    times = group[
        "time"
    ].values

    # --------------------------------------------------------
    # 统计该轨迹距离计数线最近的位置
    # --------------------------------------------------------

    distances = np.array([
        distance_to_line(p)
        for p in points
    ])

    min_distance = (
        float(distances.min())
        if len(distances) > 0
        else 99999
    )

    min_index = int(
        distances.argmin()
    )

    min_point = points[
        min_index
    ]

    # --------------------------------------------------------
    # 正负侧
    # --------------------------------------------------------

    sides = np.array([
        point_side(p)
        for p in points
    ])

    side_signs = np.sign(
        sides
    )

    # 去除恰好等于0的情况
    for i in range(
        1,
        len(side_signs)
    ):

        if side_signs[i] == 0:

            side_signs[i] = (
                side_signs[i - 1]
            )

    crossing_found = False

    candidate_count = 0

    for i in range(
        1,
        len(points)
    ):

        p_prev = points[i - 1]
        p_curr = points[i]

        frame_prev = int(
            frames[i - 1]
        )

        frame_curr = int(
            frames[i]
        )

        frame_gap = (
            frame_curr
            -
            frame_prev
        )

        # ----------------------------------------------------
        # 允许最大5帧断裂
        # ----------------------------------------------------

        if frame_gap > MAX_FRAME_GAP:
            continue

        movement = np.linalg.norm(
            p_curr - p_prev
        )

        if movement < MIN_MOVEMENT_DISTANCE:
            continue

        side_prev = (
            sides[i - 1]
        )

        side_curr = (
            sides[i]
        )

        # ----------------------------------------------------
        # 是否发生真正的侧面变化
        # ----------------------------------------------------

        if (
            side_prev * side_curr
            >= 0
        ):
            continue

        cross_point, ratio = (
            interpolate_crossing(
                p_prev,
                p_curr,
                side_prev,
                side_curr
            )
        )

        candidate_count += 1

        # ----------------------------------------------------
        # 穿线点必须在线段附近
        # ----------------------------------------------------

        if not near_line_segment(
            cross_point,
            margin=100
        ):
            continue

        # ----------------------------------------------------
        # 判断方向
        #
        # 负侧 -> 正侧 = IN
        # 正侧 -> 负侧 = OUT
        # ----------------------------------------------------

        if (
            side_prev < 0
            and
            side_curr > 0
        ):

            direction = "IN"

        elif (
            side_prev > 0
            and
            side_curr < 0
        ):

            direction = "OUT"

        else:
            continue

        # ----------------------------------------------------
        # 穿线时间
        # ----------------------------------------------------

        cross_frame = (
            frame_prev
            +
            ratio *
            frame_gap
        )

        cross_time = (
            float(times[i - 1])
            +
            ratio *
            (
                float(times[i])
                -
                float(times[i - 1])
            )
        )

        # ----------------------------------------------------
        # 后续确认
        #
        # 不再要求30px，
        # 只要求至少CONFIRM_DISTANCE
        # ----------------------------------------------------

        confirmed = False

        confirm_index = None

        for j in range(
            i,
            len(points)
        ):

            future_frame_gap = (
                int(frames[j])
                -
                frame_curr
            )

            if future_frame_gap > 60:
                break

            future_side = point_side(
                points[j]
            )

            future_distance = (
                distance_to_line(
                    points[j]
                )
            )

            if direction == "IN":

                if (
                    future_side > 0
                    and
                    future_distance
                    >= CONFIRM_DISTANCE
                ):

                    confirmed = True
                    confirm_index = j
                    break

            elif direction == "OUT":

                if (
                    future_side < 0
                    and
                    future_distance
                    >= CONFIRM_DISTANCE
                ):

                    confirmed = True
                    confirm_index = j
                    break

        # ----------------------------------------------------
        # 保存候选诊断
        # ----------------------------------------------------

        debug_records.append({

            "global_id":
                int(global_id),

            "min_distance":
                min_distance,

            "min_x":
                float(min_point[0]),

            "min_y":
                float(min_point[1]),

            "candidate":
                True,

            "direction":
                direction,

            "cross_frame":
                cross_frame,

            "cross_time":
                cross_time,

            "cross_x":
                float(cross_point[0]),

            "cross_y":
                float(cross_point[1]),

            "confirmed":
                confirmed,

            "candidate_count":
                candidate_count
        })

        if not confirmed:
            continue

        # ----------------------------------------------------
        # 找到一个有效事件
        # ----------------------------------------------------

        confirm_point = points[
            confirm_index
        ]

        confirm_frame = int(
            frames[confirm_index]
        )

        confirm_time = float(
            times[confirm_index]
        )

        confirm_distance = (
            distance_to_line(
                confirm_point
            )
        )

        count_events.append({

            "global_id":
                int(global_id),

            "direction":
                direction,

            "cross_frame":
                float(cross_frame),

            "cross_time":
                float(cross_time),

            "cross_x":
                float(cross_point[0]),

            "cross_y":
                float(cross_point[1]),

            "confirm_frame":
                confirm_frame,

            "confirm_time":
                confirm_time,

            "confirm_x":
                float(confirm_point[0]),

            "confirm_y":
                float(confirm_point[1]),

            "confirm_distance":
                float(confirm_distance)
        })

        crossing_found = True

        break

    # --------------------------------------------------------
    # 没有候选穿线，也保存诊断
    # --------------------------------------------------------

    if not crossing_found:

        debug_records.append({

            "global_id":
                int(global_id),

            "min_distance":
                min_distance,

            "min_x":
                float(min_point[0]),

            "min_y":
                float(min_point[1]),

            "candidate":
                False,

            "direction":
                "",

            "cross_frame":
                -1,

            "cross_time":
                -1,

            "cross_x":
                -1,

            "cross_y":
                -1,

            "confirmed":
                False,

            "candidate_count":
                candidate_count
        })


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


debug_df = pd.DataFrame(
    debug_records
)

debug_df = debug_df.sort_values(
    "global_id"
)

debug_df.to_csv(
    COUNT_DEBUG_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


IN_COUNT = int(
    (
        count_df["direction"]
        ==
        "IN"
    ).sum()
)

OUT_COUNT = int(
    (
        count_df["direction"]
        ==
        "OUT"
    ).sum()
)

TOTAL_COUNT = (
    IN_COUNT
    +
    OUT_COUNT
)


print("\n进出计数完成")
print(
    f"IN：{IN_COUNT}"
)
print(
    f"OUT：{OUT_COUNT}"
)
print(
    f"TOTAL：{TOTAL_COUNT}"
)

print(
    f"计数诊断："
    f"{COUNT_DEBUG_OUTPUT}"
)


# ============================================================
# 9. ROI
# ============================================================

roi_np = np.array(
    roi_points,
    dtype=np.int32
)


def point_in_roi(x, y):

    result = cv2.pointPolygonTest(
        roi_np,
        (
            float(x),
            float(y)
        ),
        False
    )

    return result >= 0


# ============================================================
# 10. ROI停留分析
# ============================================================

print(
    "\n开始区域停留分析..."
)


dwell_events = []


for global_id, group in (
    df_valid.groupby("global_id")
):

    group = group.sort_values(
        "frame"
    ).reset_index(drop=True)

    inside = False

    enter_frame = None
    enter_time = None

    last_inside_frame = None
    last_inside_time = None

    last_observed_frame = None

    visit_index = 0

    # 最近一次离开ROI的时间
    outside_start_frame = None

    for i, row in group.iterrows():

        frame = int(
            row["frame"]
        )

        time = float(
            row["time"]
        )

        x = float(
            row["cx_smooth"]
        )

        y = float(
            row["cy_smooth"]
        )

        current_inside = (
            point_in_roi(x, y)
        )

        # ----------------------------------------------------
        # 轨迹断裂
        # ----------------------------------------------------

        if last_observed_frame is not None:

            frame_gap = (
                frame
                -
                last_observed_frame
            )

            if frame_gap > ROI_MAX_GAP:

                # 如果正在ROI中
                if inside:

                    dwell_time = (
                        last_inside_time
                        -
                        enter_time
                    )

                    # 只保留有效停留
                    if dwell_time >= MIN_DWELL_TIME:

                        visit_index += 1

                        dwell_events.append({

                            "global_id":
                                int(global_id),

                            "visit_id":
                                visit_index,

                            "enter_frame":
                                enter_frame,

                            "enter_time":
                                enter_time,

                            "exit_frame":
                                last_inside_frame,

                            "exit_time":
                                last_inside_time,

                            "dwell_time":
                                dwell_time,

                            "valid":
                                True
                        })

                    inside = False

                    enter_frame = None
                    enter_time = None

        # ----------------------------------------------------
        # ROI外 -> ROI内
        # ----------------------------------------------------

        if (
            not inside
            and
            current_inside
        ):

            inside = True

            enter_frame = frame
            enter_time = time

            last_inside_frame = frame
            last_inside_time = time

        # ----------------------------------------------------
        # ROI内 -> ROI外
        # ----------------------------------------------------

        elif (
            inside
            and
            not current_inside
        ):

            dwell_time = (
                last_inside_time
                -
                enter_time
            )

            if dwell_time >= MIN_DWELL_TIME:

                visit_index += 1

                dwell_events.append({

                    "global_id":
                        int(global_id),

                    "visit_id":
                        visit_index,

                    "enter_frame":
                        enter_frame,

                    "enter_time":
                        enter_time,

                    "exit_frame":
                        frame,

                    "exit_time":
                        time,

                    "dwell_time":
                        dwell_time,

                    "valid":
                        True
                })

            inside = False

            enter_frame = None
            enter_time = None

        # ----------------------------------------------------
        # ROI内部持续
        # ----------------------------------------------------

        if (
            inside
            and
            current_inside
        ):

            last_inside_frame = frame
            last_inside_time = time

        last_observed_frame = frame

    # --------------------------------------------------------
    # 视频结束
    # --------------------------------------------------------

    if inside:

        dwell_time = (
            last_inside_time
            -
            enter_time
        )

        if dwell_time >= MIN_DWELL_TIME:

            visit_index += 1

            dwell_events.append({

                "global_id":
                    int(global_id),

                "visit_id":
                    visit_index,

                "enter_frame":
                    enter_frame,

                "enter_time":
                    enter_time,

                "exit_frame":
                    last_inside_frame,

                "exit_time":
                    last_inside_time,

                "dwell_time":
                    dwell_time,

                "valid":
                    True
            })


# ============================================================
# 11. 保存ROI结果
# ============================================================

dwell_df = pd.DataFrame(
    dwell_events
)

if len(dwell_df) == 0:

    dwell_df = pd.DataFrame(
        columns=[
            "global_id",
            "visit_id",
            "enter_frame",
            "enter_time",
            "exit_frame",
            "exit_time",
            "dwell_time",
            "valid"
        ]
    )


dwell_df.to_csv(
    DWELL_EVENT_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


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
    dwell_summary[
        "global_id"
    ].nunique()
)

REGION_VISITS = int(
    len(dwell_df)
)


if len(dwell_df) > 0:

    AVG_DWELL = float(
        dwell_df[
            "dwell_time"
        ].mean()
    )

    MAX_DWELL = float(
        dwell_df[
            "dwell_time"
        ].max()
    )

    TOTAL_DWELL = float(
        dwell_df[
            "dwell_time"
        ].sum()
    )

else:

    AVG_DWELL = 0
    MAX_DWELL = 0
    TOTAL_DWELL = 0


print("\n区域分析完成")
print(
    f"区域访问人数："
    f"{REGION_VISITORS}"
)

print(
    f"有效访问次数："
    f"{REGION_VISITS}"
)

print(
    f"平均停留："
    f"{AVG_DWELL:.2f}s"
)

print(
    f"最长停留："
    f"{MAX_DWELL:.2f}s"
)

print(
    f"总停留："
    f"{TOTAL_DWELL:.2f}s"
)


# ============================================================
# 12. 热力图
# ============================================================

print(
    "\n开始生成热力图..."
)


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
# 13. 视频信息
# ============================================================

if not os.path.exists(
    video_path
):
    raise FileNotFoundError(
        f"找不到视频："
        f"{video_path}"
    )


cap = cv2.VideoCapture(
    video_path
)

if not cap.isOpened():

    raise RuntimeError(
        "视频打开失败"
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

fps = cap.get(
    cv2.CAP_PROP_FPS
)

if fps <= 0:
    fps = 30

total_frames = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)


print(
    f"\n视频："
    f"{width} × {height}"
)

print(
    f"FPS：{fps:.2f}"
)

print(
    f"总帧数：{total_frames}"
)


# ============================================================
# 14. 热力图
# ============================================================

heatmap = np.zeros(
    (
        height,
        width
    ),
    dtype=np.float32
)


for _, row in (
    heatmap_points.iterrows()
):

    x = int(
        round(
            row["cx_smooth"]
        )
    )

    y = int(
        round(
            row["cy_smooth"]
        )
    )

    if (
        0 <= x < width
        and
        0 <= y < height
    ):

        heatmap[
            y,
            x
        ] += 1


heatmap_blur = cv2.GaussianBlur(
    heatmap,
    (
        HEATMAP_BLUR_KERNEL,
        HEATMAP_BLUR_KERNEL
    ),
    0
)


if heatmap_blur.max() > 0:

    heatmap_norm = cv2.normalize(
        heatmap_blur,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(
        np.uint8
    )

else:

    heatmap_norm = np.zeros(
        (
            height,
            width
        ),
        dtype=np.uint8
    )


heatmap_color = cv2.applyColorMap(
    heatmap_norm,
    cv2.COLORMAP_JET
)


# ============================================================
# 15. 保存静态热力图
# ============================================================

cap.set(
    cv2.CAP_PROP_POS_FRAMES,
    0
)

ret, first_frame = (
    cap.read()
)

if not ret:

    raise RuntimeError(
        "无法读取第一帧"
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
    tuple(
        map(int, P1)
    ),
    tuple(
        map(int, P2)
    ),
    (255, 255, 255),
    3
)


cv2.imwrite(
    HEATMAP_IMAGE_OUTPUT,
    heatmap_static
)


print(
    f"热力图："
    f"{HEATMAP_IMAGE_OUTPUT}"
)


# ============================================================
# 16. 建立帧索引
# ============================================================

frame_tracks = defaultdict(list)


for _, row in (
    df_valid.iterrows()
):

    frame_id = int(
        row["frame"]
    )

    frame_tracks[
        frame_id
    ].append({

        "global_id":
            int(row["global_id"]),

        "x":
            float(row["cx_smooth"]),

        "y":
            float(row["cy_smooth"])
    })


# ============================================================
# 17. 输出视频
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    VIDEO_OUTPUT,
    fourcc,
    fps,
    (
        width,
        height
    )
)


trail_history = defaultdict(list)

TRAIL_LENGTH = 30


# ============================================================
# 18. 统一分析视频
# ============================================================

print(
    "\n开始生成统一分析视频..."
)


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

    current_people = (
        frame_tracks.get(
            frame_number,
            []
        )
    )

    # --------------------------------------------------------
    # 更新轨迹
    # --------------------------------------------------------

    for person in current_people:

        gid = person[
            "global_id"
        ]

        point = (
            int(person["x"]),
            int(person["y"])
        )

        trail_history[
            gid
        ].append(point)

        if len(
            trail_history[gid]
        ) > TRAIL_LENGTH:

            trail_history[
                gid
            ] = trail_history[
                gid
            ][-TRAIL_LENGTH:]


    # --------------------------------------------------------
    # 热力图
    # --------------------------------------------------------

    display = cv2.addWeighted(
        frame,
        0.65,
        heatmap_color,
        0.35,
        0
    )


    # --------------------------------------------------------
    # ROI
    # --------------------------------------------------------

    cv2.polylines(
        display,
        [roi_np],
        True,
        (255, 255, 255),
        3
    )


    # --------------------------------------------------------
    # 计数线
    # --------------------------------------------------------

    cv2.line(
        display,
        tuple(
            map(int, P1)
        ),
        tuple(
            map(int, P2)
        ),
        (255, 255, 255),
        4
    )


    cv2.circle(
        display,
        tuple(
            map(int, P1)
        ),
        7,
        (255, 255, 255),
        -1
    )

    cv2.circle(
        display,
        tuple(
            map(int, P2)
        ),
        7,
        (255, 255, 255),
        -1
    )


    # --------------------------------------------------------
    # 轨迹
    # --------------------------------------------------------

    for gid, points in (
        trail_history.items()
    ):

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

    for person in current_people:

        gid = person[
            "global_id"
        ]

        x = int(
            person["x"]
        )

        y = int(
            person["y"]
        )

        inside = point_in_roi(
            x,
            y
        )

        if inside:
            roi_people += 1

        cv2.circle(
            display,
            (x, y),
            5,
            (0, 255, 0)
            if inside
            else (255, 255, 0),
            -1
        )

        cv2.putText(
            display,
            f"GID {gid}",
            (
                x + 5,
                y - 5
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )


    # --------------------------------------------------------
    # 信息框
    # --------------------------------------------------------

    overlay = display.copy()

    cv2.rectangle(
        overlay,
        (15, 15),
        (430, 190),
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


    current_time = (
        frame_number
        /
        fps
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
        (160, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        display,
        f"TOTAL: {TOTAL_COUNT}",
        (290, 75),
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
        f"Visitors: {REGION_VISITORS}",
        (30, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        display,
        f"Avg Dwell: {AVG_DWELL:.2f}s",
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
            f"{frame_number}/"
            f"{total_frames}"
        )


# ============================================================
# 19. 释放
# ============================================================

cap.release()
writer.release()


# ============================================================
# 20. 保存统计JSON
# ============================================================

summary = {

    "version": "V9.2",

    "video": video_path,

    "video_info": {

        "width":
            width,

        "height":
            height,

        "fps":
            fps,

        "total_frames":
            total_frames
    },

    "trajectory": {

        "trajectory_points":
            int(len(df_valid)),

        "global_ids":
            int(
                df_valid[
                    "global_id"
                ].nunique()
            )
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

            "p1":
                list(line_p1),

            "p2":
                list(line_p2)
        },

        "roi":
            [
                list(p)
                for p in roi_points
            ],

        "parameters":
            params,

        "v9_2_parameters": {

            "roi_max_gap":
                ROI_MAX_GAP,

            "min_dwell_time":
                MIN_DWELL_TIME
        }
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
# 21. 最终结果
# ============================================================

print("\n")
print("=" * 72)
print("V9.2 统一分析完成")
print("=" * 72)

print(
    f"IN：{IN_COUNT}"
)

print(
    f"OUT：{OUT_COUNT}"
)

print(
    f"TOTAL：{TOTAL_COUNT}"
)

print(
    f"\n区域访问人数："
    f"{REGION_VISITORS}"
)

print(
    f"区域访问次数："
    f"{REGION_VISITS}"
)

print(
    f"平均停留："
    f"{AVG_DWELL:.2f}s"
)

print(
    f"最长停留："
    f"{MAX_DWELL:.2f}s"
)

print("\n输出目录：")
print(OUTPUT_DIR)

print("\n核心文件：")
print(COUNT_OUTPUT)
print(COUNT_DEBUG_OUTPUT)
print(DWELL_EVENT_OUTPUT)
print(DWELL_SUMMARY_OUTPUT)
print(HEATMAP_IMAGE_OUTPUT)
print(VIDEO_OUTPUT)

print("=" * 72)