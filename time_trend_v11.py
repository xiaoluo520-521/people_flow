import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# V11 时间段人流趋势分析
# ============================================================

PROJECT_DIR = r"E:\people_flow"

RESULT_DIR = os.path.join(
    PROJECT_DIR,
    "results",
    "unified_v9_4"
)

COUNT_FILE = os.path.join(
    RESULT_DIR,
    "count_events_v9_4.csv"
)

OUTPUT_CSV = os.path.join(
    RESULT_DIR,
    "time_trend_v11.csv"
)

OUTPUT_PNG = os.path.join(
    RESULT_DIR,
    "time_trend_v11.png"
)


# ============================================================
# 参数
# ============================================================

# 每多少秒统计一次
TIME_INTERVAL = 5


# ============================================================
# 读取数据
# ============================================================

if not os.path.exists(COUNT_FILE):

    raise FileNotFoundError(
        f"找不到计数数据：{COUNT_FILE}"
    )


df = pd.read_csv(
    COUNT_FILE
)


print("=" * 70)
print("V11 时间段人流趋势分析")
print("=" * 70)

print(
    f"计数事件：{len(df)}"
)


# ============================================================
# 检查字段
# ============================================================

print()
print("数据字段：")
print(
    list(df.columns)
)


if "time" not in df.columns:

    raise ValueError(
        "count_events_v9_4.csv 中没有 time 字段"
    )


# ============================================================
# 时间转换
# ============================================================

df["time"] = pd.to_numeric(
    df["time"],
    errors="coerce"
)

df = df.dropna(
    subset=["time"]
)


# ============================================================
# 创建时间区间
# ============================================================

df["time_start"] = (
    df["time"] // TIME_INTERVAL
) * TIME_INTERVAL


df["time_end"] = (
    df["time_start"]
    + TIME_INTERVAL
)


df["time_period"] = (
    df["time_start"].astype(int).astype(str)
    + "-"
    + df["time_end"].astype(int).astype(str)
    + "s"
)


# ============================================================
# IN / OUT 统计
# ============================================================

if "direction" not in df.columns:

    raise ValueError(
        "计数文件中没有 direction 字段"
    )


trend = pd.crosstab(
    df["time_period"],
    df["direction"]
)


# 保证 IN / OUT 都存在
if "IN" not in trend.columns:

    trend["IN"] = 0


if "OUT" not in trend.columns:

    trend["OUT"] = 0


# 按时间排序
trend["_sort"] = [
    int(x.split("-")[0])
    for x in trend.index
]


trend = trend.sort_values(
    "_sort"
)


trend = trend.drop(
    columns="_sort"
)


trend["TOTAL"] = (
    trend["IN"]
    + trend["OUT"]
)


# ============================================================
# 保存 CSV
# ============================================================

trend.to_csv(
    OUTPUT_CSV,
    encoding="utf-8-sig"
)


# ============================================================
# 绘制趋势图
# ============================================================

plt.figure(
    figsize=(10, 5)
)

x = range(
    len(trend)
)

plt.plot(
    x,
    trend["IN"],
    marker="o",
    label="IN"
)

plt.plot(
    x,
    trend["OUT"],
    marker="o",
    label="OUT"
)

plt.xticks(
    x,
    trend.index,
    rotation=30
)

plt.xlabel(
    "Time Period"
)

plt.ylabel(
    "People Count"
)

plt.title(
    "People Flow Trend"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUTPUT_PNG,
    dpi=150
)

plt.close()


# ============================================================
# 输出
# ============================================================

print()
print("时间段统计：")
print(
    trend
)

print()
print("CSV：")
print(
    OUTPUT_CSV
)

print()
print("趋势图：")
print(
    OUTPUT_PNG
)

print("=" * 70)
print("时间趋势分析完成")
print("=" * 70)