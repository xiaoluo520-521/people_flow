import os
import cv2
import json
import numpy as np
import pandas as pd
from collections import defaultdict, deque


# ============================================================
# 1. 基础路径
# ============================================================

BASE_DIR = r"E:\people_flow"

CONFIG_FILE = os.path.join(
    BASE_DIR,
    "config.json"
)

TRAJECTORY_FILE = os.path.join(
    BASE_DIR,
    r"results\trajectory_reconnect_v2\reconnected_tracks.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    r"results\unified_v9_4"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. 输出文件
# ============================================================

COUNT_FILE = os.path.join(
    OUTPUT_DIR,
    "count_events_v9_4.csv"
)

COUNT_DEBUG_FILE = os.path.join(
    OUTPUT_DIR,
    "count_debug_v9_4.csv"
)

DWELL_FILE = os.path.join(
    OUTPUT_DIR,
    "dwell_events_v9_4.csv"
)

DWELL_SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "dwell_summary_v9_4.csv"
)

HEATMAP_DATA_FILE = os.path.join(
    OUTPUT_DIR,
    "heatmap_data_v9_4.csv"
)

HEATMAP_IMAGE_FILE = os.path.join(
    OUTPUT_DIR,
    "heatmap_final_v9_4.png"
)

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "analysis_summary_v9_4.json"
)

OUTPUT_VIDEO = os.path.join(
    OUTPUT_DIR,
    "analysis_result_v9_4.mp4"
)


# ============================================================
# 3. 读取配置
# ============================================================

print("=" * 70)
print("人流进出计数与区域热度分析系统 V9.4")
print("=" * 70)

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

VIDEO_FILE = config["video"]

line_config = config["counting_line"]

P1 = tuple(
    line_config["p1"]
)

P2 = tuple(
    line_config["p2"]
)

ROI = np.array(
    config["roi"],
    dtype=np.int32
)

params = config.get(
    "parameters",
    {}
)

MIN_TRACK_FRAMES = params.get(
    "min_track_frames",
    10
)

CONFIRM_DISTANCE = params.get(
    "confirm_distance",
    20
)

MIN_MOVEMENT_DISTANCE = params.get(
    "min_movement_distance",
    10
)

MAX_FRAME_GAP = params.get(
    "max_frame_gap",
    10
)

HEATMAP_BLUR_KERNEL = params.get(
    "heatmap_blur_kernel",
    51
)

print("\n配置读取成功")

print(
    f"视频：{VIDEO_FILE}"
)

print(
    f"计数线：{P1} -> {P2}"
)

print(
    f"ROI点数：{len(ROI)}"
)

print(
    f"最小轨迹长度：{MIN_TRACK_FRAMES}"
)

print(
    f"最大帧间隔：{MAX_FRAME_GAP}"
)


# ============================================================
# 4. 读取轨迹
# ============================================================

print("\n读取 Global ID 轨迹...")

df = pd.read_csv(
    TRAJECTORY_FILE
)

if "cx_smooth" in df.columns:

    X_COL = "cx_smooth"
    Y_COL = "cy_smooth"

else:

    X_COL = "cx"
    Y_COL = "cy"


required = [
    "global_id",
    "frame",
    "time",
    X_COL,
    Y_COL
]

for col in required:

    if col not in df.columns:

        raise ValueError(
            f"缺少字段：{col}"
        )

df = df.sort_values(
    ["global_id", "frame"]
).reset_index(
    drop=True
)

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
    f"轨迹点：{len(df)}"
)

print(
    f"Global ID：{df['global_id'].nunique()}"
)

print(
    f"有效 Global ID：{len(valid_ids)}"
)


# ============================================================
# 5. 几何函数
# ============================================================

def cross_value(
    point,
    p1,
    p2
):

    x, y = point

    x1, y1 = p1
    x2, y2 = p2

    return (
        (x2 - x1) * (y - y1)
        -
        (y2 - y1) * (x - x1)
    )


def get_side(point):

    value = cross_value(
        point,
        P1,
        P2
    )

    if value > 0:
        return 1

    if value < 0:
        return -1

    return 0


def point_line_distance(point):

    x, y = point

    x1, y1 = P1
    x2, y2 = P2

    numerator = abs(
        (y2 - y1) * x
        -
        (x2 - x1) * y
        +
        x2 * y1
        -
        y2 * x1
    )

    denominator = np.sqrt(
        (y2 - y1) ** 2
        +
        (x2 - x1) ** 2
    )

    if denominator == 0:

        return 999999

    return numerator / denominator


def segment_intersection(
    a,
    b,
    p,
    q
):

    def orientation(
        a,
        b,
        c
    ):

        value = (
            (b[0] - a[0])
            *
            (c[1] - a[1])
            -
            (b[1] - a[1])
            *
            (c[0] - a[0])
        )

        if abs(value) < 1e-8:

            return 0

        return (
            1
            if value > 0
            else -1
        )

    o1 = orientation(
        a,
        b,
        p
    )

    o2 = orientation(
        a,
        b,
        q
    )

    o3 = orientation(
        p,
        q,
        a
    )

    o4 = orientation(
        p,
        q,
        b
    )

    return (
        o1 != o2
        and
        o3 != o4
    )


# ============================================================
# 6. 进出计数
# ============================================================

print("\n开始进出计数分析...")

events = []

debug_records = []

counted_ids = set()


for gid, group in df_valid.groupby(
    "global_id"
):

    group = group.sort_values(
        "frame"
    ).reset_index(
        drop=True
    )

    points = []

    for _, row in group.iterrows():

        points.append({

            "frame":
                int(row["frame"]),

            "time":
                float(row["time"]),

            "x":
                float(row[X_COL]),

            "y":
                float(row[Y_COL])

        })


    min_distance = 999999

    min_frame = None
    min_x = None
    min_y = None

    for p in points:

        d = point_line_distance(
            (
                p["x"],
                p["y"]
            )
        )

        if d < min_distance:

            min_distance = d

            min_frame = p["frame"]
            min_x = p["x"]
            min_y = p["y"]


    candidate = False
    direction = None

    cross_frame = None
    cross_time = None
    cross_x = None
    cross_y = None


    for i in range(
        1,
        len(points)
    ):

        prev = points[i - 1]
        curr = points[i]

        frame_gap = (
            curr["frame"]
            -
            prev["frame"]
        )

        if frame_gap > MAX_FRAME_GAP:

            continue


        prev_point = (
            prev["x"],
            prev["y"]
        )

        curr_point = (
            curr["x"],
            curr["y"]
        )

        side_prev = get_side(
            prev_point
        )

        side_curr = get_side(
            curr_point
        )

        if side_prev == 0:
            continue

        if side_curr == 0:
            continue

        if side_prev == side_curr:
            continue


        if not segment_intersection(
            prev_point,
            curr_point,
            P1,
            P2
        ):

            continue


        # ----------------------------------------------------
        # 穿线前后运动距离
        # ----------------------------------------------------

        before_start = max(
            0,
            i - 8
        )

        after_end = min(
            len(points),
            i + 9
        )

        if (
            i - before_start < 2
            or
            after_end - (i + 1) < 2
        ):

            continue


        before_points = points[
            before_start:i
        ]

        after_points = points[
            i + 1:after_end
        ]


        bx = (
            before_points[-1]["x"]
            -
            before_points[0]["x"]
        )

        by = (
            before_points[-1]["y"]
            -
            before_points[0]["y"]
        )

        ax = (
            after_points[-1]["x"]
            -
            after_points[0]["x"]
        )

        ay = (
            after_points[-1]["y"]
            -
            after_points[0]["y"]
        )


        before_distance = np.sqrt(
            bx * bx
            +
            by * by
        )

        after_distance = np.sqrt(
            ax * ax
            +
            ay * ay
        )


        if (
            before_distance
            <
            MIN_MOVEMENT_DISTANCE
        ):

            continue

        if (
            after_distance
            <
            MIN_MOVEMENT_DISTANCE
        ):

            continue


        # ----------------------------------------------------
        # 判断方向
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


        candidate = True

        cross_frame = curr["frame"]
        cross_time = curr["time"]
        cross_x = curr["x"]
        cross_y = curr["y"]

        break


    # ========================================================
    # 诊断记录
    # ========================================================

    debug_records.append({

        "global_id":
            gid,

        "track_frames":
            len(points),

        "min_distance":
            min_distance,

        "min_frame":
            min_frame,

        "min_x":
            min_x,

        "min_y":
            min_y,

        "candidate":
            candidate,

        "direction":
            direction,

        "cross_frame":
            cross_frame,

        "cross_time":
            cross_time,

        "cross_x":
            cross_x,

        "cross_y":
            cross_y

    })


    # ========================================================
    # 保存计数事件
    # ========================================================

    if candidate:

        if gid in counted_ids:

            continue

        counted_ids.add(gid)

        events.append({

            "global_id":
                gid,

            "direction":
                direction,

            "frame":
                cross_frame,

            "time":
                cross_time,

            "x":
                cross_x,

            "y":
                cross_y,

            "distance":
                min_distance

        })

        print(
            f"{direction:<3} "
            f"GID={gid:<4} "
            f"time={cross_time:.2f}s "
            f"distance={min_distance:.2f}px"
        )


# ============================================================
# 7. 保存计数
# ============================================================

events_df = pd.DataFrame(
    events
)

debug_df = pd.DataFrame(
    debug_records
)

events_df.to_csv(
    COUNT_FILE,
    index=False,
    encoding="utf-8-sig"
)

debug_df.to_csv(
    COUNT_DEBUG_FILE,
    index=False,
    encoding="utf-8-sig"
)

if len(events_df) > 0:

    in_count = (
        events_df["direction"]
        ==
        "IN"
    ).sum()

    out_count = (
        events_df["direction"]
        ==
        "OUT"
    ).sum()

else:

    in_count = 0
    out_count = 0

total_count = (
    in_count
    +
    out_count
)

print("\n进出计数完成")

print(
    f"IN：{in_count}"
)

print(
    f"OUT：{out_count}"
)

print(
    f"TOTAL：{total_count}"
)


# ============================================================
# 8. 区域停留分析
# ============================================================

print("\n开始区域停留分析...")

dwell_events = []

summary_records = []


for gid, group in df_valid.groupby(
    "global_id"
):

    group = group.sort_values(
        "frame"
    ).reset_index(
        drop=True
    )

    inside = False

    enter_frame = None
    enter_time = None

    visit_index = 0

    dwell_times = []


    for _, row in group.iterrows():

        x = float(
            row[X_COL]
        )

        y = float(
            row[Y_COL]
        )

        frame = int(
            row["frame"]
        )

        time = float(
            row["time"]
        )

        inside_now = (
            cv2.pointPolygonTest(
                ROI,
                (
                    x,
                    y
                ),
                False
            )
            >= 0
        )


        if (
            inside_now
            and
            not inside
        ):

            inside = True

            enter_frame = frame
            enter_time = time

            visit_index += 1


        elif (
            not inside_now
            and
            inside
        ):

            inside = False

            exit_frame = frame
            exit_time = time

            dwell_time = (
                exit_time
                -
                enter_time
            )

            if dwell_time >= 0.5:

                dwell_times.append(
                    dwell_time
                )

                dwell_events.append({

                    "global_id":
                        gid,

                    "visit":
                        visit_index,

                    "enter_frame":
                        enter_frame,

                    "enter_time":
                        enter_time,

                    "exit_frame":
                        exit_frame,

                    "exit_time":
                        exit_time,

                    "dwell_time":
                        dwell_time

                })


    # 如果视频结束时还在 ROI
    if inside:

        exit_frame = int(
            group.iloc[-1]["frame"]
        )

        exit_time = float(
            group.iloc[-1]["time"]
        )

        dwell_time = (
            exit_time
            -
            enter_time
        )

        if dwell_time >= 0.5:

            dwell_times.append(
                dwell_time
            )

            dwell_events.append({

                "global_id":
                    gid,

                "visit":
                    visit_index,

                "enter_frame":
                    enter_frame,

                "enter_time":
                    enter_time,

                "exit_frame":
                    exit_frame,

                "exit_time":
                    exit_time,

                "dwell_time":
                    dwell_time

            })


    if len(dwell_times) > 0:

        summary_records.append({

            "global_id":
                gid,

            "visits":
                len(dwell_times),

            "total_dwell":
                sum(dwell_times),

            "avg_dwell":
                np.mean(
                    dwell_times
                ),

            "max_dwell":
                max(
                    dwell_times
                )

        })


dwell_df = pd.DataFrame(
    dwell_events
)

summary_df = pd.DataFrame(
    summary_records
)

dwell_df.to_csv(
    DWELL_FILE,
    index=False,
    encoding="utf-8-sig"
)

summary_df.to_csv(
    DWELL_SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig"
)


if len(summary_df) > 0:

    region_visitors = (
        summary_df["global_id"]
        .nunique()
    )

    total_visits = (
        summary_df["visits"]
        .sum()
    )

    total_dwell = (
        summary_df["total_dwell"]
        .sum()
    )

    avg_dwell = (
        dwell_df["dwell_time"]
        .mean()
    )

    max_dwell = (
        dwell_df["dwell_time"]
        .max()
    )

else:

    region_visitors = 0
    total_visits = 0
    total_dwell = 0
    avg_dwell = 0
    max_dwell = 0


print(
    f"区域访问人数：{region_visitors}"
)

print(
    f"区域访问次数：{total_visits}"
)

print(
    f"平均停留：{avg_dwell:.2f}s"
)

print(
    f"最长停留：{max_dwell:.2f}s"
)


# ============================================================
# 9. 热力图数据
# ============================================================

print("\n开始生成热力图...")

heatmap_df = df_valid[
    [
        "global_id",
        "frame",
        "time",
        X_COL,
        Y_COL
    ]
].copy()

heatmap_df = heatmap_df.rename(
    columns={
        X_COL: "x",
        Y_COL: "y"
    }
)

heatmap_df.to_csv(
    HEATMAP_DATA_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 10. 打开视频
# ============================================================

cap = cv2.VideoCapture(
    VIDEO_FILE
)

if not cap.isOpened():

    raise RuntimeError(
        "无法打开视频"
    )

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

total_frames = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)

print(
    f"视频：{width} × {height}"
)

print(
    f"FPS：{fps:.2f}"
)

print(
    f"总帧数：{total_frames}"
)


# ============================================================
# 11. 生成静态热力图
# ============================================================

heatmap = np.zeros(
    (
        height,
        width
    ),
    dtype=np.float32
)


for _, row in heatmap_df.iterrows():

    x = int(
        round(row["x"])
    )

    y = int(
        round(row["y"])
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


kernel = HEATMAP_BLUR_KERNEL

if kernel % 2 == 0:

    kernel += 1

heatmap = cv2.GaussianBlur(
    heatmap,
    (
        kernel,
        kernel
    ),
    0
)


if heatmap.max() > 0:

    heatmap_norm = cv2.normalize(
        heatmap,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

else:

    heatmap_norm = heatmap


heatmap_uint8 = (
    heatmap_norm
    .astype(np.uint8)
)

heatmap_color = cv2.applyColorMap(
    heatmap_uint8,
    cv2.COLORMAP_JET
)

heatmap_color = cv2.cvtColor(
    heatmap_color,
    cv2.COLOR_BGR2RGB
)

cv2.imwrite(
    HEATMAP_IMAGE_FILE,
    cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_RGB2BGR
    )
)


# ============================================================
# 12. 生成统一分析视频
# ============================================================

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


# 当前帧轨迹
frame_tracks = defaultdict(list)

for _, row in df_valid.iterrows():

    frame_tracks[
        int(row["frame"])
    ].append(
        (
            int(row["global_id"]),
            float(row[X_COL]),
            float(row[Y_COL])
        )
    )


# 计数事件
event_dict = {}

for _, row in events_df.iterrows():

    event_dict[
        int(row["global_id"])
    ] = row.to_dict()


trail_history = defaultdict(
    lambda: deque(
        maxlen=40
    )
)


frame_index = 0


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_index += 1


    # ========================================================
    # 当前轨迹
    # ========================================================

    current_tracks = frame_tracks.get(
        frame_index,
        []
    )


    for gid, x, y in current_tracks:

        trail_history[
            gid
        ].append(
            (
                int(x),
                int(y)
            )
        )


    # ========================================================
    # 绘制热力图
    # ========================================================

    if heatmap_color is not None:

        overlay = cv2.resize(
            heatmap_color,
            (
                width,
                height
            )
        )

        overlay = cv2.cvtColor(
            overlay,
            cv2.COLOR_RGB2BGR
        )

        frame = cv2.addWeighted(
            frame,
            0.65,
            overlay,
            0.35,
            0
        )


    # ========================================================
    # ROI
    # ========================================================

    cv2.polylines(
        frame,
        [
            ROI
        ],
        True,
        (0, 255, 255),
        3
    )

    cv2.putText(
        frame,
        "ROI",
        tuple(
            ROI[0]
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )


    # ========================================================
    # 绘制轨迹
    # ========================================================

    for gid, x, y in current_tracks:

        trail = list(
            trail_history[gid]
        )

        for j in range(
            1,
            len(trail)
        ):

            cv2.line(
                frame,
                trail[j - 1],
                trail[j],
                (255, 255, 0),
                2
            )


        point = (
            int(x),
            int(y)
        )

        cv2.circle(
            frame,
            point,
            5,
            (0, 255, 255),
            -1
        )

        cv2.putText(
            frame,
            f"GID {gid}",
            (
                point[0] + 8,
                point[1] - 8
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )


    # ========================================================
    # 计数线
    # ========================================================

    cv2.line(
        frame,
        P1,
        P2,
        (0, 0, 255),
        4
    )

    cv2.putText(
        frame,
        "COUNTING LINE",
        (
            P1[0],
            P1[1] - 15
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )


    # ========================================================
    # 标记计数人员
    # ========================================================

    for gid, info in event_dict.items():

        if frame_index < info["frame"]:
            continue

        for current_gid, x, y in current_tracks:

            if current_gid != gid:
                continue

            point = (
                int(x),
                int(y)
            )

            if info["direction"] == "IN":

                cv2.circle(
                    frame,
                    point,
                    13,
                    (0, 255, 0),
                    3
                )

                cv2.putText(
                    frame,
                    "IN",
                    (
                        point[0] + 15,
                        point[1]
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            else:

                cv2.circle(
                    frame,
                    point,
                    13,
                    (255, 0, 0),
                    3
                )

                cv2.putText(
                    frame,
                    "OUT",
                    (
                        point[0] + 15,
                        point[1]
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2
                )


    # ========================================================
    # 信息面板
    # ========================================================

    cv2.rectangle(
        frame,
        (20, 20),
        (360, 180),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"IN    : {in_count}",
        (35, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"OUT   : {out_count}",
        (35, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 100, 0),
        2
    )

    cv2.putText(
        frame,
        f"TOTAL : {total_count}",
        (35, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"ROI visitors: {region_visitors}",
        (35, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )


    writer.write(
        frame
    )


cap.release()
writer.release()


# ============================================================
# 13. 保存总体分析结果
# ============================================================

summary = {

    "video":
        VIDEO_FILE,

    "resolution":
        [
            width,
            height
        ],

    "fps":
        fps,

    "total_frames":
        total_frames,

    "global_ids":
        int(
            df["global_id"]
            .nunique()
        ),

    "valid_global_ids":
        int(
            len(valid_ids)
        ),

    "in_count":
        int(in_count),

    "out_count":
        int(out_count),

    "total_count":
        int(total_count),

    "region_visitors":
        int(region_visitors),

    "region_visits":
        int(total_visits),

    "average_dwell":
        float(avg_dwell),

    "max_dwell":
        float(max_dwell),

    "total_dwell":
        float(total_dwell),

    "counting_line":
        {
            "p1":
                list(P1),

            "p2":
                list(P2)
        },

    "roi":
        ROI.tolist()

}


with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        ensure_ascii=False,
        indent=4
    )


print("\n" + "=" * 70)
print("V9.4 统一分析完成")
print("=" * 70)

print(
    f"IN：{in_count}"
)

print(
    f"OUT：{out_count}"
)

print(
    f"TOTAL：{total_count}"
)

print(
    f"ROI访问人数：{region_visitors}"
)

print(
    f"ROI访问次数：{total_visits}"
)

print(
    f"平均停留：{avg_dwell:.2f}s"
)

print(
    f"最长停留：{max_dwell:.2f}s"
)

print(
    f"\n输出目录：{OUTPUT_DIR}"
)

print("=" * 70)