import os
import cv2
import pandas as pd


# ============================================================
# V11 Global ID 轨迹回放
# ============================================================

PROJECT_DIR = r"E:\people_flow"

VIDEO_PATH = os.path.join(
    PROJECT_DIR,
    "videos",
    "test.mp4"
)

TRACK_FILE = os.path.join(
    PROJECT_DIR,
    "results",
    "trajectory_reconnect_v2",
    "reconnected_tracks.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "results",
    "trajectory_replay_v11"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 读取轨迹
# ============================================================

df = pd.read_csv(
    TRACK_FILE
)


print("=" * 70)
print("V11 Global ID 轨迹回放")
print("=" * 70)

print(
    f"Global ID 数量：{df['global_id'].nunique()}"
)


# ============================================================
# 显示可选 Global ID
# ============================================================

global_ids = sorted(
    df["global_id"].unique()
)


print()
print("可用 Global ID：")

print(
    global_ids
)


# ============================================================
# 用户选择
# ============================================================

while True:

    try:

        gid = int(
            input(
                "\n请输入要回放的 Global ID："
            )
        )

    except ValueError:

        print("请输入整数。")
        continue

    if gid in global_ids:

        break

    print(
        "该 Global ID 不存在，请重新输入。"
    )


# ============================================================
# 获取目标轨迹
# ============================================================

target = df[
    df["global_id"] == gid
].copy()


target = target.sort_values(
    "frame"
)


print()
print(
    f"GID {gid} 轨迹点：{len(target)}"
)

print(
    f"起始帧：{target['frame'].min()}"
)

print(
    f"结束帧：{target['frame'].max()}"
)


# ============================================================
# 视频信息
# ============================================================

cap = cv2.VideoCapture(
    VIDEO_PATH
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


# ============================================================
# 输出视频
# ============================================================

output_file = os.path.join(
    OUTPUT_DIR,
    f"trajectory_replay_GID{gid}.mp4"
)


fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)


writer = cv2.VideoWriter(
    output_file,
    fourcc,
    fps,
    (width, height)
)


# ============================================================
# 转成字典，方便查找
# ============================================================

trajectory_dict = {}

for _, row in target.iterrows():

    frame_id = int(
        row["frame"]
    )

    trajectory_dict[
        frame_id
    ] = (
        float(row["cx"]),
        float(row["cy"])
    )


# ============================================================
# 开始回放
# ============================================================

frame_id = 0

trail = []

MAX_TRAIL_LENGTH = 100


while True:

    ret, frame = cap.read()

    if not ret:

        break


    # --------------------------------------------------------
    # 当前帧是否存在目标
    # --------------------------------------------------------

    if frame_id in trajectory_dict:

        cx, cy = trajectory_dict[
            frame_id
        ]

        point = (
            int(cx),
            int(cy)
        )

        trail.append(
            point
        )

        if len(trail) > MAX_TRAIL_LENGTH:

            trail.pop(0)


    # --------------------------------------------------------
    # 绘制轨迹
    # --------------------------------------------------------

    for i in range(
        1,
        len(trail)
    ):

        cv2.line(
            frame,
            trail[i - 1],
            trail[i],
            (0, 255, 255),
            3
        )


    # --------------------------------------------------------
    # 当前目标
    # --------------------------------------------------------

    if frame_id in trajectory_dict:

        cx, cy = trajectory_dict[
            frame_id
        ]

        cx = int(cx)
        cy = int(cy)

        cv2.circle(
            frame,
            (cx, cy),
            8,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            frame,
            f"GID {gid}",
            (cx + 10, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    # --------------------------------------------------------
    # 顶部信息
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (0, 0),
        (420, 70),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"Global ID: {gid}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Frame: {frame_id}",
        (15, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    writer.write(
        frame
    )


    frame_id += 1


# ============================================================
# 释放
# ============================================================

cap.release()

writer.release()


print()
print("=" * 70)
print("轨迹回放完成")
print("=" * 70)

print()
print("Global ID：", gid)

print(
    "轨迹点：",
    len(target)
)

print()
print("输出视频：")

print(
    output_file
)

print("=" * 70)