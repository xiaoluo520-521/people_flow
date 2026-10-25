import pandas as pd
import numpy as np
import os
import cv2
import math

# ============================================================
# 轨迹清洗 V1
# 功能：
# 1. 删除过短轨迹
# 2. 检测异常跳点
# 3. 对轨迹进行平滑
# 4. 输出 cleaned_tracks.csv
# 5. 输出轨迹可视化视频
# ============================================================


# =========================
# 1. 参数配置
# =========================

INPUT_CSV = r"E:\people_flow\results\trajectory\tracks.csv"

OUTPUT_DIR = r"E:\people_flow\results\trajectory_clean"

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "cleaned_tracks.csv"
)

OUTPUT_VIDEO = os.path.join(
    OUTPUT_DIR,
    "cleaned_trajectory.mp4"
)

VIDEO_PATH = r"E:\people_flow\videos\test.mp4"


# 最少轨迹长度
# 小于等于这个数量的轨迹直接删除
MIN_TRACK_LENGTH = 5

# 最大允许单帧位移
# 如果相邻两帧中心点移动超过这个距离，
# 就认为可能存在异常跳点
MAX_JUMP_DISTANCE = 100

# 平滑窗口
SMOOTH_WINDOW = 5


# =========================
# 2. 创建输出目录
# =========================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# =========================
# 3. 读取轨迹
# =========================

print("==============================")
print("轨迹清洗系统 V1")
print("==============================")

print("读取轨迹文件：")
print(INPUT_CSV)

df = pd.read_csv(INPUT_CSV)

print()
print("原始轨迹数据：")
print(f"数据量：{len(df)}")
print(f"Track ID数量：{df['track_id'].nunique()}")


# =========================
# 4. 删除过短轨迹
# =========================

track_counts = df.groupby("track_id").size()

valid_ids = track_counts[
    track_counts > MIN_TRACK_LENGTH
].index

df_clean = df[
    df["track_id"].isin(valid_ids)
].copy()

removed_ids = set(
    df["track_id"].unique()
) - set(valid_ids)


print()
print("==============================")
print("步骤1：短轨迹过滤")
print("==============================")

print(
    f"原始ID数量：{df['track_id'].nunique()}"
)

print(
    f"删除轨迹数量：{len(removed_ids)}"
)

print(
    f"保留轨迹数量：{df_clean['track_id'].nunique()}"
)

print(
    f"删除数据点数量：{len(df) - len(df_clean)}"
)


# =========================
# 5. 异常跳点检测
# =========================

print()
print("==============================")
print("步骤2：异常跳点检测")
print("==============================")

df_clean = df_clean.sort_values(
    ["track_id", "frame"]
).reset_index(drop=True)


# 用来记录异常点
df_clean["is_jump"] = False


jump_count = 0


for track_id, group in df_clean.groupby("track_id"):

    indices = group.index.tolist()

    xs = group["cx"].values
    ys = group["cy"].values

    for i in range(1, len(indices)):

        dx = xs[i] - xs[i - 1]
        dy = ys[i] - ys[i - 1]

        distance = math.sqrt(
            dx * dx + dy * dy
        )

        if distance > MAX_JUMP_DISTANCE:

            df_clean.loc[
                indices[i],
                "is_jump"
            ] = True

            jump_count += 1


print(
    f"检测到异常跳点：{jump_count}"
)


# =========================
# 6. 删除异常跳点
# =========================

df_clean = df_clean[
    ~df_clean["is_jump"]
].copy()

df_clean.drop(
    columns=["is_jump"],
    inplace=True
)


# =========================
# 7. 轨迹平滑
# =========================

print()
print("==============================")
print("步骤3：轨迹平滑")
print("==============================")


def smooth_track(group):

    group = group.sort_values("frame").copy()

    group["cx_smooth"] = (
        group["cx"]
        .rolling(
            window=SMOOTH_WINDOW,
            center=True,
            min_periods=1
        )
        .median()
    )

    group["cy_smooth"] = (
        group["cy"]
        .rolling(
            window=SMOOTH_WINDOW,
            center=True,
            min_periods=1
        )
        .median()
    )

    return group


df_clean = (
    df_clean
    .groupby(
        "track_id",
        group_keys=False
    )
    .apply(smooth_track)
    .reset_index(drop=True)
)


# =========================
# 8. 保存清洗后的 CSV
# =========================

df_clean.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)


print()
print("==============================")
print("步骤4：保存清洗结果")
print("==============================")

print(
    f"清洗后数据量：{len(df_clean)}"
)

print(
    f"清洗后ID数量：{df_clean['track_id'].nunique()}"
)

print()
print(
    f"CSV：{OUTPUT_CSV}"
)


# =========================
# 9. 生成轨迹可视化视频
# =========================

print()
print("==============================")
print("步骤5：生成轨迹视频")
print("==============================")


cap = cv2.VideoCapture(
    VIDEO_PATH
)

if not cap.isOpened():

    print("无法打开视频：")
    print(VIDEO_PATH)

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
    (width, height)
)


# =========================
# 10. 按 Track ID 保存轨迹
# =========================

trajectory_dict = {}

for track_id, group in df_clean.groupby("track_id"):

    points = []

    for _, row in group.iterrows():

        points.append(
            (
                int(row["cx_smooth"]),
                int(row["cy_smooth"])
            )
        )

    trajectory_dict[track_id] = points


# =========================
# 11. 重新读取视频并绘制轨迹
# =========================

frame_idx = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_idx += 1

    # 当前帧所有轨迹
    current_df = df_clean[
        df_clean["frame"] == frame_idx
    ]

    for _, row in current_df.iterrows():

        track_id = int(
            row["track_id"]
        )

        cx = int(
            row["cx_smooth"]
        )

        cy = int(
            row["cy_smooth"]
        )

        # =========================
        # 绘制最近轨迹
        # =========================

        group = df_clean[
            (df_clean["track_id"] == track_id)
            &
            (df_clean["frame"] <= frame_idx)
            &
            (df_clean["frame"] >= frame_idx - 50)
        ]

        points = [
            (
                int(x),
                int(y)
            )
            for x, y in zip(
                group["cx_smooth"],
                group["cy_smooth"]
            )
        ]

        for i in range(
            1,
            len(points)
        ):

            cv2.line(
                frame,
                points[i - 1],
                points[i],
                (0, 255, 0),
                2
            )

        # =========================
        # 绘制中心点
        # =========================

        cv2.circle(
            frame,
            (cx, cy),
            4,
            (0, 0, 255),
            -1
        )

        # =========================
        # 绘制ID
        # =========================

        cv2.putText(
            frame,
            f"ID:{track_id}",
            (
                cx + 5,
                cy - 5
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    # 左上角显示信息

    cv2.putText(
        frame,
        f"Frame: {frame_idx}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    writer.write(frame)


cap.release()
writer.release()


print()
print("==============================")
print("轨迹清洗完成！")
print("==============================")

print(
    f"清洗后CSV：{OUTPUT_CSV}"
)

print(
    f"轨迹视频：{OUTPUT_VIDEO}"
)

print("==============================")