import os
import json
import tkinter as tk
from tkinter import ttk, messagebox

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk


# ============================================================
# V10 Dashboard
# ============================================================

PROJECT_DIR = r"E:\people_flow"

RESULT_DIR = os.path.join(
    PROJECT_DIR,
    "results",
    "unified_v9_4"
)

SUMMARY_FILE = os.path.join(
    RESULT_DIR,
    "analysis_summary_v9_4.json"
)

COUNT_FILE = os.path.join(
    RESULT_DIR,
    "count_events_v9_4.csv"
)

DWELL_FILE = os.path.join(
    RESULT_DIR,
    "dwell_events_v9_4.csv"
)

HEATMAP_FILE = os.path.join(
    RESULT_DIR,
    "heatmap_final_v9_4.png"
)

VIDEO_FILE = os.path.join(
    RESULT_DIR,
    "analysis_result_v9_4.mp4"
)


# ============================================================
# matplotlib 中文
# ============================================================

plt.rcParams["font.sans-serif"] = [
    "SimHei",
    "Microsoft YaHei",
    "Arial Unicode MS"
]

plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 读取 JSON
# ============================================================

if not os.path.exists(SUMMARY_FILE):

    raise FileNotFoundError(
        "没有找到分析结果：\n"
        + SUMMARY_FILE
    )


with open(
    SUMMARY_FILE,
    "r",
    encoding="utf-8"
) as f:

    summary = json.load(f)


# ============================================================
# 读取计数数据
# ============================================================

count_df = None

if os.path.exists(COUNT_FILE):

    count_df = pd.read_csv(
        COUNT_FILE
    )


# ============================================================
# 读取停留数据
# ============================================================

dwell_df = None

if os.path.exists(DWELL_FILE):

    dwell_df = pd.read_csv(
        DWELL_FILE
    )


# ============================================================
# 获取统计值
# ============================================================

def get_value(data, *keys, default=0):

    for key in keys:

        if key in data:

            return data[key]

    return default


global_ids = get_value(
    summary,
    "global_ids",
    "global_id_count",
    default=0
)

valid_ids = get_value(
    summary,
    "valid_global_ids",
    default=0
)

in_count = get_value(
    summary,
    "in",
    "IN",
    "in_count",
    default=0
)

out_count = get_value(
    summary,
    "out",
    "OUT",
    "out_count",
    default=0
)

total_count = get_value(
    summary,
    "total",
    "TOTAL",
    "total_count",
    default=in_count + out_count
)

roi_visitors = get_value(
    summary,
    "roi_visitors",
    "region_visitors",
    default=0
)

roi_visits = get_value(
    summary,
    "roi_visits",
    "region_visits",
    default=0
)

avg_dwell = get_value(
    summary,
    "avg_dwell",
    "average_dwell",
    "avg_dwell_time",
    default=0
)

max_dwell = get_value(
    summary,
    "max_dwell",
    "max_dwell_time",
    default=0
)


# ============================================================
# 主窗口
# ============================================================

root = tk.Tk()

root.title(
    "人流进出计数与区域热度分析系统 V10"
)

root.geometry(
    "1450x900"
)

root.minsize(
    1200,
    750
)


# ============================================================
# 标题
# ============================================================

title = tk.Label(
    root,
    text="人流进出计数与区域热度分析系统",
    font=("Microsoft YaHei", 24, "bold")
)

title.pack(
    pady=(15, 5)
)

subtitle = tk.Label(
    root,
    text="YOLO11 + ByteTrack + Global ID + ROI + Heatmap",
    font=("Microsoft YaHei", 11)
)

subtitle.pack(
    pady=(0, 10)
)


# ============================================================
# KPI 区域
# ============================================================

kpi_frame = tk.Frame(root)

kpi_frame.pack(
    fill="x",
    padx=20,
    pady=10
)


def create_kpi(
    parent,
    title,
    value,
    column
):

    frame = tk.Frame(
        parent,
        relief="solid",
        bd=1,
        padx=20,
        pady=10
    )

    frame.grid(
        row=0,
        column=column,
        padx=8,
        sticky="nsew"
    )

    parent.grid_columnconfigure(
        column,
        weight=1
    )

    label = tk.Label(
        frame,
        text=title,
        font=("Microsoft YaHei", 11)
    )

    label.pack()

    value_label = tk.Label(
        frame,
        text=str(value),
        font=("Arial", 22, "bold")
    )

    value_label.pack(
        pady=(5, 0)
    )

    return value_label


create_kpi(
    kpi_frame,
    "Global ID",
    global_ids,
    0
)

create_kpi(
    kpi_frame,
    "有效轨迹",
    valid_ids,
    1
)

create_kpi(
    kpi_frame,
    "进场 IN",
    in_count,
    2
)

create_kpi(
    kpi_frame,
    "出场 OUT",
    out_count,
    3
)

create_kpi(
    kpi_frame,
    "总计",
    total_count,
    4
)

create_kpi(
    kpi_frame,
    "ROI访问人数",
    roi_visitors,
    5
)


# ============================================================
# 第二行统计
# ============================================================

info_frame = tk.Frame(root)

info_frame.pack(
    fill="x",
    padx=20,
    pady=5
)


info_text = (
    f"ROI访问次数：{roi_visits}    |    "
    f"平均停留：{avg_dwell:.2f}s    |    "
    f"最长停留：{max_dwell:.2f}s"
)

info_label = tk.Label(
    info_frame,
    text=info_text,
    font=("Microsoft YaHei", 13)
)

info_label.pack()


# ============================================================
# 中间区域
# ============================================================

middle_frame = tk.Frame(root)

middle_frame.pack(
    fill="both",
    expand=True,
    padx=20,
    pady=10
)


# ============================================================
# 左侧：热力图
# ============================================================

heatmap_frame = tk.LabelFrame(
    middle_frame,
    text="人流热力图",
    font=("Microsoft YaHei", 12, "bold")
)

heatmap_frame.pack(
    side="left",
    fill="both",
    expand=True,
    padx=(0, 8)
)


heatmap_label = tk.Label(
    heatmap_frame
)

heatmap_label.pack(
    fill="both",
    expand=True,
    padx=10,
    pady=10
)


def load_heatmap():

    if not os.path.exists(HEATMAP_FILE):

        heatmap_label.config(
            text="热力图文件不存在"
        )

        return

    image = Image.open(
        HEATMAP_FILE
    )

    max_width = 650
    max_height = 430

    image.thumbnail(
        (max_width, max_height)
    )

    photo = ImageTk.PhotoImage(
        image
    )

    heatmap_label.config(
        image=photo
    )

    heatmap_label.image = photo


load_heatmap()


# ============================================================
# 右侧：柱状图 + 趋势
# ============================================================

chart_frame = tk.Frame(
    middle_frame
)

chart_frame.pack(
    side="right",
    fill="both",
    expand=True,
    padx=(8, 0)
)


# ============================================================
# IN / OUT 柱状图
# ============================================================

fig1, ax1 = plt.subplots(
    figsize=(5.5, 3.2)
)

ax1.bar(
    ["IN", "OUT"],
    [in_count, out_count]
)

ax1.set_title(
    "进出人数统计"
)

ax1.set_ylabel(
    "人数"
)

ax1.grid(
    axis="y",
    alpha=0.3
)

fig1.tight_layout()


canvas1 = FigureCanvasTkAgg(
    fig1,
    master=chart_frame
)

canvas1.draw()

canvas1.get_tk_widget().pack(
    fill="both",
    expand=True
)


# ============================================================
# 时间趋势
# ============================================================

fig2, ax2 = plt.subplots(
    figsize=(5.5, 3.2)
)


def create_time_trend():

    if (
        count_df is None
        or len(count_df) == 0
    ):

        ax2.text(
            0.5,
            0.5,
            "暂无计数数据",
            ha="center",
            va="center"
        )

        return

    if "time" not in count_df.columns:

        ax2.text(
            0.5,
            0.5,
            "缺少 time 字段",
            ha="center",
            va="center"
        )

        return

    df = count_df.copy()

    df["time"] = pd.to_numeric(
        df["time"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["time"]
    )

    if len(df) == 0:

        return

    # 5 秒一个时间段
    df["time_bin"] = (
        df["time"] // 5
    ) * 5

    if "direction" in df.columns:

        trend = (
            df.groupby(
                ["time_bin", "direction"]
            )
            .size()
            .unstack(
                fill_value=0
            )
        )

        if "IN" in trend.columns:

            ax2.plot(
                trend.index,
                trend["IN"],
                marker="o",
                label="IN"
            )

        if "OUT" in trend.columns:

            ax2.plot(
                trend.index,
                trend["OUT"],
                marker="o",
                label="OUT"
            )

    else:

        trend = (
            df.groupby("time_bin")
            .size()
        )

        ax2.plot(
            trend.index,
            trend.values,
            marker="o"
        )

    ax2.set_title(
        "时间段人流趋势"
    )

    ax2.set_xlabel(
        "时间 / 秒"
    )

    ax2.set_ylabel(
        "人数"
    )

    ax2.legend()

    ax2.grid(
        alpha=0.3
    )


create_time_trend()

fig2.tight_layout()


canvas2 = FigureCanvasTkAgg(
    fig2,
    master=chart_frame
)

canvas2.draw()

canvas2.get_tk_widget().pack(
    fill="both",
    expand=True
)


# ============================================================
# 底部信息
# ============================================================

bottom_frame = tk.Frame(root)

bottom_frame.pack(
    fill="x",
    padx=20,
    pady=(0, 15)
)


video_status = (
    "分析视频：已生成"
    if os.path.exists(VIDEO_FILE)
    else "分析视频：未找到"
)

video_label = tk.Label(
    bottom_frame,
    text=video_status,
    font=("Microsoft YaHei", 10)
)

video_label.pack(
    side="left"
)


result_label = tk.Label(
    bottom_frame,
    text=f"结果目录：{RESULT_DIR}",
    font=("Microsoft YaHei", 9)
)

result_label.pack(
    side="right"
)


# ============================================================
# 刷新按钮
# ============================================================

def refresh():

    messagebox.showinfo(
        "提示",
        "当前 Dashboard 使用的是最近一次 V9.4 分析结果。\n\n"
        "如果修改了计数线或 ROI，请重新运行 main_v9_4.py，"
        "然后重新打开 Dashboard。"
    )


refresh_button = tk.Button(
    root,
    text="刷新提示",
    command=refresh,
    font=("Microsoft YaHei", 10),
    padx=15,
    pady=5
)

refresh_button.pack(
    pady=(0, 10)
)


# ============================================================
# 启动
# ============================================================

root.mainloop()