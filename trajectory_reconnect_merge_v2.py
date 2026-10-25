import pandas as pd
import numpy as np
import os


# ============================================================
# 配置
# ============================================================

INPUT_CSV = (
    r"E:\people_flow\results\trajectory_clean"
    r"\cleaned_tracks.csv"
)

CANDIDATE_CSV = (
    r"E:\people_flow\results\trajectory_reconnect_v2"
    r"\reconnect_candidates.csv"
)

OUTPUT_DIR = (
    r"E:\people_flow\results\trajectory_reconnect_v2"
)

MATCH_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "reconnect_matches.csv"
)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "reconnected_tracks.csv"
)


# ============================================================
# 参数
# ============================================================

# 最低匹配分数
MIN_SCORE = 0.60

# 每个旧ID最多连接一个新ID
# 每个新ID最多连接一个旧ID


# ============================================================
# 评分函数
# ============================================================

def calculate_score(gap, distance, direction_similarity):

    # --------------------------------------------------------
    # 时间评分
    # gap 越小越好
    # --------------------------------------------------------

    time_score = max(
        0,
        1 - gap / 15
    )

    # --------------------------------------------------------
    # 空间评分
    # distance 越小越好
    # --------------------------------------------------------

    distance_score = max(
        0,
        1 - distance / 100
    )

    # --------------------------------------------------------
    # 方向评分
    # cosine similarity [-1, 1]
    # 转换到 [0, 1]
    # --------------------------------------------------------

    direction_score = (
        direction_similarity + 1
    ) / 2

    # --------------------------------------------------------
    # 综合评分
    # --------------------------------------------------------

    score = (
        0.20 * time_score
        +
        0.35 * distance_score
        +
        0.35 * direction_score
        +
        0.10 * (
            time_score * distance_score
        )
    )

    return score


# ============================================================
# 主程序
# ============================================================

print("=" * 70)
print("轨迹 ID 自动重连 V2.2")
print("=" * 70)


# ============================================================
# 创建目录
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 读取轨迹
# ============================================================

df = pd.read_csv(
    INPUT_CSV
)

print(
    f"原始数据量：{len(df)}"
)

print(
    f"原始 Track ID：{df['track_id'].nunique()}"
)


# ============================================================
# 读取候选
# ============================================================

candidates = pd.read_csv(
    CANDIDATE_CSV
)

print(
    f"候选重连数量：{len(candidates)}"
)


# ============================================================
# 计算综合评分
# ============================================================

scores = []

for _, row in candidates.iterrows():

    score = calculate_score(
        gap=float(row["gap"]),
        distance=float(row["distance"]),
        direction_similarity=float(
            row["direction_similarity"]
        )
    )

    scores.append(score)


candidates["score"] = scores

candidates["score"] = candidates[
    "score"
].round(4)


# ============================================================
# 按综合评分排序
# ============================================================

candidates = candidates.sort_values(
    by="score",
    ascending=False
).reset_index(drop=True)


# ============================================================
# 一对一匹配
# ============================================================

used_a = set()
used_b = set()

matches = []

for _, row in candidates.iterrows():

    id_a = int(row["id_a"])
    id_b = int(row["id_b"])

    score = float(row["score"])

    # --------------------------------------------------------
    # 最低评分过滤
    # --------------------------------------------------------

    if score < MIN_SCORE:
        continue

    # --------------------------------------------------------
    # 一对一约束
    # --------------------------------------------------------

    if id_a in used_a:
        continue

    if id_b in used_b:
        continue

    # --------------------------------------------------------
    # 接受匹配
    # --------------------------------------------------------

    used_a.add(id_a)
    used_b.add(id_b)

    matches.append({
        "old_id": id_a,
        "new_id": id_b,
        "gap": int(row["gap"]),
        "distance": float(
            row["distance"]
        ),
        "direction_similarity":
            float(
                row["direction_similarity"]
            ),
        "score": score
    })


# ============================================================
# 保存匹配结果
# ============================================================

match_df = pd.DataFrame(
    matches
)

if len(match_df) > 0:

    match_df = match_df.sort_values(
        by="score",
        ascending=False
    )

match_df.to_csv(
    MATCH_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 打印匹配结果
# ============================================================

print()
print("=" * 70)
print("最终接受的 ID 重连")
print("=" * 70)

print(
    f"成功重连数量：{len(match_df)}"
)

if len(match_df) > 0:

    print()

    print(
        match_df.to_string(
            index=False
        )
    )

else:

    print(
        "没有满足条件的重连。"
    )


# ============================================================
# 建立 Global ID
# ============================================================

# 所有原始 ID
all_ids = sorted(
    df["track_id"].unique()
)


# ------------------------------------------------------------
# 初始化：
# 每个原始 ID 默认就是自己的 Global ID
# ------------------------------------------------------------

parent = {
    int(track_id): int(track_id)
    for track_id in all_ids
}


# ============================================================
# Union-Find
# ============================================================

def find(x):

    while parent[x] != x:

        parent[x] = parent[
            parent[x]
        ]

        x = parent[x]

    return x


def union(a, b):

    root_a = find(a)
    root_b = find(b)

    if root_a != root_b:

        # 使用较小 ID 作为 Global ID
        if root_a < root_b:

            parent[root_b] = root_a

        else:

            parent[root_a] = root_b


# ============================================================
# 合并轨迹
# ============================================================

for _, row in match_df.iterrows():

    old_id = int(row["old_id"])
    new_id = int(row["new_id"])

    union(
        old_id,
        new_id
    )


# ============================================================
# 生成 Global ID
# ============================================================

df["global_id"] = df[
    "track_id"
].apply(
    lambda x: find(int(x))
)


# ============================================================
# 重新排序
# ============================================================

df = df.sort_values(
    by=[
        "global_id",
        "frame"
    ]
)


# ============================================================
# 保存最终轨迹
# ============================================================

df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 统计
# ============================================================

raw_id_count = df[
    "track_id"
].nunique()

global_id_count = df[
    "global_id"
].nunique()


print()
print("=" * 70)
print("ID 重连完成")
print("=" * 70)

print(
    f"原始 Track ID：{raw_id_count}"
)

print(
    f"Global ID：{global_id_count}"
)

print(
    f"减少 ID 数量："
    f"{raw_id_count - global_id_count}"
)

print()
print(
    f"匹配结果：{MATCH_OUTPUT}"
)

print(
    f"最终轨迹：{OUTPUT_CSV}"
)

print("=" * 70)