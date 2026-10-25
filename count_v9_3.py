import os
import cv2
import numpy as np
import pandas as pd
from collections import defaultdict, deque


# ============================================================
# 1. 路径配置
# ============================================================

BASE_DIR = r"E:\people_flow"

TRAJECTORY_FILE = os.path.join(
    BASE_DIR,
    r"results\trajectory_reconnect_v2\reconnected_tracks.csv"
)

VIDEO_FILE = os.path.join(
    BASE_DIR,
    r"videos\test.mp4"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    r"results\count_v9_3"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. 计数线
# ============================================================
# 先使用你之前验证成功的水平线
#
# P1 ---------------------------- P2
#
# 后面可以换成你实际场景中的计数线
#
P1 = (200, 360)
P2 = (1100, 360)


# ============================================================
# 3. 参数
# ============================================================

MIN_TRACK_FRAMES = 10

# 两个轨迹点之间允许的最大帧间隔
MAX_FRAME_GAP = 10

# 穿线前后用于计算运动方向的点数
DIRECTION_WINDOW = 8

# 穿线附近允许的线段扩展范围
SEGMENT_MARGIN = 80

# 穿线后至少移动多少像素才确认
CONFIRM_DISTANCE = 20

# 最小运动距离
MIN_MOVEMENT_DISTANCE = 10

# 一个 Global ID 只允许计数一次
COUNT_ONCE = True

# 轨迹显示长度
TRAIL_LENGTH = 40


# ============================================================
# 4. 输出文件
# ============================================================

EVENT_FILE = os.path.join(
    OUTPUT_DIR,
    "count_events_v9_3.csv"
)

DEBUG_FILE = os.path.join(
    OUTPUT_DIR,
    "count_debug_v9_3.csv"
)

OUTPUT_VIDEO = os.path.join(
    OUTPUT_DIR,
    "count_result_v9_3.mp4"
)


# ============================================================
# 5. 读取轨迹
# ============================================================

print("=" * 70)
print("Global ID 鲁棒进出计数 V9.3")
print("=" * 70)

print("\n读取轨迹文件...")

df = pd.read_csv(TRAJECTORY_FILE)

print(f"轨迹数据：{len(df)}")

# 兼容不同字段
if "cx_smooth" in df.columns:
    X_COL = "cx_smooth"
else:
    X_COL = "cx"

if "cy_smooth" in df.columns:
    Y_COL = "cy_smooth"
else:
    Y_COL = "cy"

required_cols = [
    "global_id",
    "frame",
    "time",
    X_COL,
    Y_COL
]

for col in required_cols:
    if col not in df.columns:
        raise ValueError(
            f"轨迹文件缺少字段：{col}"
        )

df = df.sort_values(
    ["global_id", "frame"]
).reset_index(drop=True)

print(
    f"Global ID：{df['global_id'].nunique()}"
)


# ============================================================
# 6. 轨迹过滤
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
    f"有效 Global ID：{len(valid_ids)}"
)


# ============================================================
# 7. 几何函数
# ============================================================

def cross_value(point, p1, p2):
    """
    计算点相对于有向直线 P1 -> P2 的叉积。

    < 0：一侧
    > 0：另一侧
    = 0：在线上
    """

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

    elif value < 0:
        return -1

    return 0


def point_to_line_distance(point):
    """
    点到无限延长计数线的距离
    """

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


def line_segment_intersection(
    a,
    b,
    p,
    q
):
    """
    判断线段 AB 是否与 PQ 相交。
    """

    def orientation(a, b, c):

        value = (
            (b[0] - a[0]) * (c[1] - a[1])
            -
            (b[1] - a[1]) * (c[0] - a[0])
        )

        if abs(value) < 1e-8:
            return 0

        return 1 if value > 0 else -1

    def on_segment(a, b, c):

        return (
            min(a[0], b[0]) - SEGMENT_MARGIN
            <= c[0]
            <= max(a[0], b[0]) + SEGMENT_MARGIN
            and
            min(a[1], b[1]) - SEGMENT_MARGIN
            <= c[1]
            <= max(a[1], b[1]) + SEGMENT_MARGIN
        )

    o1 = orientation(a, b, p)
    o2 = orientation(a, b, q)
    o3 = orientation(p, q, a)
    o4 = orientation(p, q, b)

    if (
        o1 != o2
        and
        o3 != o4
    ):
        return True

    if o1 == 0 and on_segment(a, b, p):
        return True

    if o2 == 0 and on_segment(a, b, q):
        return True

    if o3 == 0 and on_segment(p, q, a):
        return True

    if o4 == 0 and on_segment(p, q, b):
        return True

    return False


def distance(a, b):

    return np.sqrt(
        (a[0] - b[0]) ** 2
        +
        (a[1] - b[1]) ** 2
    )


# ============================================================
# 8. 判断轨迹段是否穿过计数线
# ============================================================

def is_valid_crossing(
    prev_point,
    curr_point
):

    # 第一层：两点必须位于计数线两侧
    side_prev = get_side(prev_point)
    side_curr = get_side(curr_point)

    if side_prev == 0:
        return False

    if side_curr == 0:
        return True

    if side_prev == side_curr:
        return False

    # 第二层：轨迹线段与计数线相交
    if not line_segment_intersection(
        prev_point,
        curr_point,
        P1,
        P2
    ):
        return False

    return True


# ============================================================
# 9. 计算穿线方向
# ============================================================

def calculate_direction(
    points_before,
    points_after
):

    if len(points_before) < 2:
        return None

    if len(points_after) < 2:
        return None

    before_start = points_before[0]
    before_end = points_before[-1]

    after_start = points_after[0]
    after_end = points_after[-1]

    vx_before = (
        before_end[0]
        -
        before_start[0]
    )

    vy_before = (
        before_end[1]
        -
        before_start[1]
    )

    vx_after = (
        after_end[0]
        -
        after_start[0]
    )

    vy_after = (
        after_end[1]
        -
        after_start[1]
    )

    movement_before = np.sqrt(
        vx_before ** 2
        +
        vy_before ** 2
    )

    movement_after = np.sqrt(
        vx_after ** 2
        +
        vy_after ** 2
    )

    if movement_before < MIN_MOVEMENT_DISTANCE:
        return None

    if movement_after < MIN_MOVEMENT_DISTANCE:
        return None

    # 用计数线两侧判断方向
    side_before = get_side(
        before_end
    )

    side_after = get_side(
        after_end
    )

    if (
        side_before < 0
        and
        side_after > 0
    ):
        return "IN"

    elif (
        side_before > 0
        and
        side_after < 0
    ):
        return "OUT"

    return None


# ============================================================
# 10. 分析每一个 Global ID
# ============================================================

events = []
debug_records = []

counted_ids = set()

print("\n开始分析穿线...")


for global_id, group in df_valid.groupby(
    "global_id"
):

    group = group.sort_values(
        "frame"
    ).reset_index(drop=True)

    points = []

    for _, row in group.iterrows():

        points.append({
            "frame": int(row["frame"]),
            "time": float(row["time"]),
            "x": float(row[X_COL]),
            "y": float(row[Y_COL])
        })

    candidate_found = False

    candidate_frame = None
    candidate_time = None
    candidate_x = None
    candidate_y = None

    candidate_direction = None

    min_distance = 999999

    min_distance_frame = None
    min_distance_x = None
    min_distance_y = None

    # --------------------------------------------------------
    # 找距离计数线最近的点
    # --------------------------------------------------------

    for p in points:

        d = point_to_line_distance(
            (p["x"], p["y"])
        )

        if d < min_distance:

            min_distance = d

            min_distance_frame = p["frame"]
            min_distance_x = p["x"]
            min_distance_y = p["y"]

    # --------------------------------------------------------
    # 检查相邻轨迹段
    # --------------------------------------------------------

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

        if not is_valid_crossing(
            prev_point,
            curr_point
        ):
            continue

        # ----------------------------------------------------
        # 找穿线前方向
        # ----------------------------------------------------

        start_before = max(
            0,
            i - DIRECTION_WINDOW
        )

        before_points = [
            (
                p["x"],
                p["y"]
            )
            for p in points[
                start_before:i
            ]
        ]

        # ----------------------------------------------------
        # 找穿线后方向
        # ----------------------------------------------------

        end_after = min(
            len(points),
            i + 1 + DIRECTION_WINDOW
        )

        after_points = [
            (
                p["x"],
                p["y"]
            )
            for p in points[
                i + 1:end_after
            ]
        ]

        direction = calculate_direction(
            before_points,
            after_points
        )

        if direction is None:
            continue

        candidate_found = True

        candidate_frame = curr["frame"]
        candidate_time = curr["time"]
        candidate_x = curr["x"]
        candidate_y = curr["y"]

        candidate_direction = direction

        break

    # --------------------------------------------------------
    # 记录诊断信息
    # --------------------------------------------------------

    debug_records.append({

        "global_id":
            global_id,

        "track_frames":
            len(points),

        "min_distance":
            min_distance,

        "min_frame":
            min_distance_frame,

        "min_x":
            min_distance_x,

        "min_y":
            min_distance_y,

        "candidate":
            candidate_found,

        "direction":
            candidate_direction,

        "confirmed":
            False,

        "candidate_frame":
            candidate_frame,

        "candidate_time":
            candidate_time,

        "candidate_x":
            candidate_x,

        "candidate_y":
            candidate_y
    })

    # --------------------------------------------------------
    # 如果发现候选
    # --------------------------------------------------------

    if candidate_found:

        if COUNT_ONCE:

            if global_id in counted_ids:
                continue

        counted_ids.add(
            global_id
        )

        events.append({

            "global_id":
                global_id,

            "direction":
                candidate_direction,

            "frame":
                candidate_frame,

            "time":
                candidate_time,

            "x":
                candidate_x,

            "y":
                candidate_y,

            "distance":
                min_distance
        })

        print(
            f"{candidate_direction:<3} "
            f"GID={global_id:<4} "
            f"time={candidate_time:.2f}s "
            f"distance={min_distance:.2f}px"
        )


# ============================================================
# 11. 保存结果
# ============================================================

events_df = pd.DataFrame(
    events
)

debug_df = pd.DataFrame(
    debug_records
)

events_df.to_csv(
    EVENT_FILE,
    index=False,
    encoding="utf-8-sig"
)

debug_df.to_csv(
    DEBUG_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 12. 统计
# ============================================================

if len(events_df) > 0:

    in_count = (
        events_df["direction"]
        == "IN"
    ).sum()

    out_count = (
        events_df["direction"]
        == "OUT"
    ).sum()

else:

    in_count = 0
    out_count = 0

total_count = (
    in_count
    +
    out_count
)

print("\n" + "=" * 70)

print("进出计数完成")

print(
    f"IN   ：{in_count}"
)

print(
    f"OUT  ：{out_count}"
)

print(
    f"TOTAL：{total_count}"
)

print("=" * 70)

print(
    f"\n事件文件：{EVENT_FILE}"
)

print(
    f"诊断文件：{DEBUG_FILE}"
)


# ============================================================
# 13. 生成可视化视频
# ============================================================

print("\n开始生成计数可视化视频...")

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
# 14. 按 Global ID 保存轨迹
# ============================================================

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


trail_history = defaultdict(
    lambda: deque(
        maxlen=TRAIL_LENGTH
    )
)


# ============================================================
# 15. 视频逐帧处理
# ============================================================

event_dict = {}

for _, row in events_df.iterrows():

    event_dict[
        int(row["global_id"])
    ] = row.to_dict()


frame_index = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_index += 1

    # --------------------------------------------------------
    # 当前帧轨迹
    # --------------------------------------------------------

    current_ids = frame_tracks.get(
        frame_index,
        []
    )

    current_id_set = set()

    for gid, x, y in current_ids:

        current_id_set.add(gid)

        point = (
            int(x),
            int(y)
        )

        trail_history[
            gid
        ].append(point)

    # --------------------------------------------------------
    # 绘制轨迹
    # --------------------------------------------------------

    for gid in current_id_set:

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
                (255, 200, 0),
                2
            )

        x, y = trail[-1]

        # Global ID
        cv2.circle(
            frame,
            (x, y),
            5,
            (0, 255, 255),
            -1
        )

        cv2.putText(
            frame,
            f"GID {gid}",
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    # --------------------------------------------------------
    # 绘制计数线
    # --------------------------------------------------------

    cv2.line(
        frame,
        P1,
        P2,
        (0, 0, 255),
        4
    )

    cv2.circle(
        frame,
        P1,
        7,
        (0, 255, 255),
        -1
    )

    cv2.circle(
        frame,
        P2,
        7,
        (0, 255, 255),
        -1
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

    # --------------------------------------------------------
    # 显示已经计数的人
    # --------------------------------------------------------

    for gid, info in event_dict.items():

        if (
            frame_index
            >=
            info["frame"]
        ):

            # 找当前轨迹点
            for current_gid, x, y in current_ids:

                if current_gid != gid:
                    continue

                x = int(x)
                y = int(y)

                if info["direction"] == "IN":

                    cv2.circle(
                        frame,
                        (x, y),
                        12,
                        (0, 255, 0),
                        3
                    )

                    label = "IN"

                else:

                    cv2.circle(
                        frame,
                        (x, y),
                        12,
                        (255, 0, 0),
                        3
                    )

                    label = "OUT"

                cv2.putText(
                    frame,
                    label,
                    (x + 15, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (
                        0,
                        255,
                        0
                    )
                    if label == "IN"
                    else (
                        255,
                        0,
                        0
                    ),
                    2
                )

    # --------------------------------------------------------
    # 信息面板
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (20, 20),
        (340, 125),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"IN   : {in_count}",
        (35, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"OUT  : {out_count}",
        (35, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 100, 0),
        2
    )

    cv2.putText(
        frame,
        f"TOTAL: {total_count}",
        (35, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # 帧号
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Frame: {frame_index}/{total_frames}",
        (
            width - 250,
            height - 20
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    writer.write(
        frame
    )


cap.release()
writer.release()

print(
    "\n可视化视频生成完成："
)

print(
    OUTPUT_VIDEO
)

print("\n" + "=" * 70)
print("V9.3 全部完成")
print("=" * 70)