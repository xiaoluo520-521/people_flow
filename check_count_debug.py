import pandas as pd

file_path = r"E:\people_flow\results\unified_v9_2\count_debug_v9_2.csv"

df = pd.read_csv(file_path)

print("=" * 70)
print("计数线诊断")
print("=" * 70)

print(f"\n总诊断 Global ID：{len(df)}")

print("\n1. 距离计数线最近的20个 Global ID：")

cols = [
    "global_id",
    "min_distance",
    "min_x",
    "min_y",
    "candidate",
    "direction",
    "confirmed",
    "candidate_count"
]

show = df.sort_values(
    "min_distance"
).head(20)

print(
    show[cols].to_string(
        index=False
    )
)

print("\n2. 有穿线候选的 Global ID：")

candidate_df = df[
    df["candidate"] == True
]

print(
    f"候选数量：{len(candidate_df)}"
)

if len(candidate_df) > 0:

    print(
        candidate_df[
            cols
        ].to_string(
            index=False
        )
    )

print("\n3. 候选方向统计：")

if len(candidate_df) > 0:

    print(
        candidate_df[
            "direction"
        ].value_counts()
    )

print("\n4. 确认成功数量：")

confirmed_df = df[
    df["confirmed"] == True
]

print(
    f"确认成功：{len(confirmed_df)}"
)

if len(confirmed_df) > 0:

    print(
        confirmed_df[
            cols
        ].to_string(
            index=False
        )
    )

print("\n5. 最接近计数线的距离统计：")

print(
    df["min_distance"].describe()
)

print("\n6. 距离计数线不同范围的数量：")

print(
    "<10 px :",
    (df["min_distance"] < 10).sum()
)

print(
    "<20 px :",
    (df["min_distance"] < 20).sum()
)

print(
    "<30 px :",
    (df["min_distance"] < 30).sum()
)

print(
    "<50 px :",
    (df["min_distance"] < 50).sum()
)

print(
    "<100 px:",
    (df["min_distance"] < 100).sum()
)

print("=" * 70)