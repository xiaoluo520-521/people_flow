import cv2
import pandas as pd
import os


# ============================================================
# 路径
# ============================================================

VIDEO_PATH = r"E:\people_flow\videos\test.mp4"

INPUT_CSV = (
    r"E:\people_flow\results\trajectory_reconnect_v2"
    r"\reconnected_tracks.csv"
)

OUTPUT_DIR = (
    r"E:\people_flow\results\trajectory_reconnect_v2"
)

OUTPUT_VIDEO = os.path.join(
    OUTPUT_DIR,
    "reconnected_trajectory.mp4"
)


# ============================================================
# 参数
# ============================================================

TRAIL_LENGTH = 40


# ============================================================
# 读取轨迹
# ============================================================

print("=" * 60)
print("Global ID 轨迹可视化 V2.3")
print("=" * 60)

df = pd.read_csv(INPUT_CSV)

print(
    f"轨迹数据量：{len(df)}"
)

print(
    f"原始 Track ID：{df['track_id'].nunique()}"
)

print(
    f"Global ID：{df['global_id'].nunique()}"
)


# ============================================================
# 视频
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
# 输出视频
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)


# ============================================================
# frame -> 数据
# ============================================================

frame_tracks = {}

for frame_id, group in df.groupby(
    "frame"
):

    frame_tracks[
        int(frame_id)
    ] = group


# ============================================================
# Global ID 轨迹缓存
# ============================================================

trails = {}


# ============================================================
# 处理视频
# ============================================================

for frame_id in range(
    1,
    total_frames + 1
):

    ret, frame = cap.read()

    if not ret:
        break


    # ========================================================
    # 当前帧轨迹
    # ========================================================

    if frame_id in frame_tracks:

        frame_data = frame_tracks[
            frame_id
        ]

        for _, row in frame_data.iterrows():

            gid = int(
                row["global_id"]
            )

            x = int(
                row["cx_smooth"]
            )

            y = int(
                row["cy_smooth"]
            )


            # ------------------------------------------------
            # 创建轨迹
            # ------------------------------------------------

            if gid not in trails:

                trails[gid] = []


            trails[gid].append(
                (x, y)
            )


            # ------------------------------------------------
            # 限制轨迹长度
            # ------------------------------------------------

            if len(
                trails[gid]
            ) > TRAIL_LENGTH:

                trails[gid].pop(0)


    # ========================================================
    # 绘制所有 Global ID
    # ========================================================

    for gid, points in trails.items():

        if len(points) < 2:
            continue


        # ----------------------------------------------------
        # 使用 OpenCV 根据 ID 自动生成颜色
        # ----------------------------------------------------

        color = (
            int((gid * 67) % 256),
            int((gid * 127) % 256),
            int((gid * 191) % 256)
        )


        # ----------------------------------------------------
        # 绘制轨迹
        # ----------------------------------------------------

        for i in range(
            1,
            len(points)
        ):

            cv2.line(
                frame,
                points[i - 1],
                points[i],
                color,
                2
            )


    # ========================================================
    # 绘制当前目标
    # ========================================================

    if frame_id in frame_tracks:

        frame_data = frame_tracks[
            frame_id
        ]

        for _, row in frame_data.iterrows():

            gid = int(
                row["global_id"]
            )

            x = int(
                row["cx_smooth"]
            )

            y = int(
                row["cy_smooth"]
            )


            color = (
                int((gid * 67) % 256),
                int((gid * 127) % 256),
                int((gid * 191) % 256)
            )


            # ------------------------------------------------
            # 中心点
            # ------------------------------------------------

            cv2.circle(
                frame,
                (x, y),
                5,
                color,
                -1
            )


            # ------------------------------------------------
            # Global ID
            # ------------------------------------------------

            cv2.putText(
                frame,
                f"GID {gid}",
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2
            )


    # ========================================================
    # 左上角信息
    # ========================================================

    cv2.rectangle(
        frame,
        (10, 10),
        (360, 85),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        frame,
        f"Frame: {frame_id}/{total_frames}",
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Global IDs: {df['global_id'].nunique()}",
        (20, 67),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # ========================================================
    # 写入
    # ========================================================

    out.write(frame)


# ============================================================
# 释放
# ============================================================

cap.release()
out.release()


print()
print("=" * 60)
print("Global ID 可视化完成")
print("=" * 60)

print(
    f"输出视频：{OUTPUT_VIDEO}"
)

print("=" * 60)