import pandas as pd
import numpy as np
import os

# ============================================================
# 参数
# ============================================================

INPUT_CSV = r"E:\people_flow\results\trajectory_clean\cleaned_tracks.csv"

# 最大允许丢失帧数
MAX_GAP = 15

# 最大空间距离
MAX_DISTANCE = 100

# 最少轨迹长度
MIN_TRACK_LENGTH = 10

# 用于估计运动方向的点数
DIRECTION_POINTS = 8


# ============================================================
# 计算两个向量的余弦相似度
# ============================================================

def cosine_similarity(v1, v2):

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0

    return np.dot(v1, v2) / (norm1 * norm2)


# ============================================================
# 计算轨迹运动方向
# ============================================================

def get_direction(track, from_start=False):

    track = track.sort_values("frame")

    if len(track) < 2:
        return np.array([0.0, 0.0])

    if from_start:

        pts = track.head(DIRECTION_POINTS)

        x1 = pts["cx_smooth"].iloc[0]
        y1 = pts["cy_smooth"].iloc[0]

        x2 = pts["cx_smooth"].iloc[-1]
        y2 = pts["cy_smooth"].iloc[-1]

    else:

        pts = track.tail(DIRECTION_POINTS)

        x1 = pts["cx_smooth"].iloc[0]
        y1 = pts["cy_smooth"].iloc[0]

        x2 = pts["cx_smooth"].iloc[-1]
        y2 = pts["cy_smooth"].iloc[-1]

    return np.array([
        x2 - x1,
        y2 - y1
    ])


# ============================================================
# 读取数据
# ============================================================

print("=" * 60)
print("轨迹 ID 重连 V2")
print("=" * 60)

df = pd.read_csv(INPUT_CSV)

print(f"轨迹数据量：{len(df)}")
print(f"原始 Track ID 数量：{df['track_id'].nunique()}")


# ============================================================
# 删除过短轨迹
# ============================================================

track_counts = df.groupby("track_id").size()

valid_ids = track_counts[
    track_counts >= MIN_TRACK_LENGTH
].index

df = df[df["track_id"].isin(valid_ids)].copy()

print(f"过滤后 Track ID 数量：{df['track_id'].nunique()}")


# ============================================================
# 建立轨迹信息
# ============================================================

tracks = {}

for track_id, group in df.groupby("track_id"):

    group = group.sort_values("frame")

    tracks[track_id] = {

        "data": group,

        "start_frame": int(group["frame"].iloc[0]),
        "end_frame": int(group["frame"].iloc[-1]),

        "start_x": float(group["cx_smooth"].iloc[0]),
        "start_y": float(group["cy_smooth"].iloc[0]),

        "end_x": float(group["cx_smooth"].iloc[-1]),
        "end_y": float(group["cy_smooth"].iloc[-1]),

        "start_direction":
            get_direction(group, from_start=True),

        "end_direction":
            get_direction(group, from_start=False)
    }


# ============================================================
# 寻找可能的 ID 重连
# ============================================================

candidates = []

track_ids = list(tracks.keys())

for id_a in track_ids:

    A = tracks[id_a]

    for id_b in track_ids:

        if id_a == id_b:
            continue

        B = tracks[id_b]

        # ----------------------------------------------------
        # A 必须在 B 前面结束
        # ----------------------------------------------------

        if A["end_frame"] >= B["start_frame"]:
            continue

        # ----------------------------------------------------
        # 时间间隔
        # ----------------------------------------------------

        gap = B["start_frame"] - A["end_frame"] - 1

        if gap < 0 or gap > MAX_GAP:
            continue

        # ----------------------------------------------------
        # 空间距离
        # ----------------------------------------------------

        dx = B["start_x"] - A["end_x"]
        dy = B["start_y"] - A["end_y"]

        distance = np.sqrt(dx * dx + dy * dy)

        if distance > MAX_DISTANCE:
            continue

        # ----------------------------------------------------
        # 运动方向一致性
        # ----------------------------------------------------

        direction_similarity = cosine_similarity(
            A["end_direction"],
            B["start_direction"]
        )

        # ----------------------------------------------------
        # 连续运动方向要求
        # ----------------------------------------------------

        if direction_similarity < 0.5:
            continue

        # ----------------------------------------------------
        # 记录候选
        # ----------------------------------------------------

        candidates.append({

            "id_a": id_a,
            "id_b": id_b,

            "end_frame_a": A["end_frame"],
            "start_frame_b": B["start_frame"],

            "gap": gap,

            "distance": round(distance, 2),

            "direction_similarity":
                round(direction_similarity, 3),

            "end_x": round(A["end_x"], 2),
            "end_y": round(A["end_y"], 2),

            "start_x": round(B["start_x"], 2),
            "start_y": round(B["start_y"], 2)
        })


# ============================================================
# 输出候选结果
# ============================================================

candidate_df = pd.DataFrame(candidates)

OUTPUT_DIR = r"E:\people_flow\results\trajectory_reconnect_v2"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "reconnect_candidates.csv"
)

candidate_df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 打印结果
# ============================================================

print()
print("=" * 60)
print("候选重连结果")
print("=" * 60)

print(f"候选数量：{len(candidate_df)}")

if len(candidate_df) > 0:

    candidate_df = candidate_df.sort_values(
        by=[
            "distance",
            "gap",
            "direction_similarity"
        ],
        ascending=[
            True,
            True,
            False
        ]
    )

    print()

    print(
        candidate_df.head(50).to_string(
            index=False
        )
    )

else:

    print("没有找到符合条件的候选轨迹。")


print()
print("=" * 60)

print(
    f"候选文件：{OUTPUT_CSV}"
)

print("=" * 60)