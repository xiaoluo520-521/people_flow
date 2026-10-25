import tkinter as tk
from tkinter import ttk, messagebox
import os
import pandas as pd
import numpy as np
import cv2
import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ============================================================
# 1. 路径配置
# ============================================================

BASE_DIR = r"E:\people_flow"

VIDEO_PATH = os.path.join(
    BASE_DIR,
    "videos",
    "test.mp4"
)

TRACK_CSV = os.path.join(
    BASE_DIR,
    "results",
    "trajectory_reconnect_v2",
    "reconnected_tracks.csv"
)

COUNT_CSV = os.path.join(
    BASE_DIR,
    "results",
    "count_global_v5_1",
    "global_count_events_v5_1.csv"
)

DWELL_EVENT_CSV = os.path.join(
    BASE_DIR,
    "results",
    "region_dwell_v6",
    "region_dwell_events_v6.csv"
)

DWELL_SUMMARY_CSV = os.path.join(
    BASE_DIR,
    "results",
    "region_dwell_v6",
    "region_dwell_summary_v6.csv"
)

HEATMAP_CSV = os.path.join(
    BASE_DIR,
    "results",
    "heatmap_v7",
    "heatmap_data_v7.csv"
)

HEATMAP_IMAGE = os.path.join(
    BASE_DIR,
    "results",
    "heatmap_v7",
    "heatmap_final_v7.png"
)


# ============================================================
# 2. 数据读取
# ============================================================

def safe_read_csv(path):

    if not os.path.exists(path):

        print(
            f"文件不存在：{path}"
        )

        return pd.DataFrame()

    try:

        return pd.read_csv(
            path
        )

    except Exception as e:

        print(
            f"读取失败：{path}"
        )

        print(e)

        return pd.DataFrame()


# ============================================================
# 3. Dashboard
# ============================================================

class Dashboard:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "人流进出计数与区域热度分析系统"
        )

        self.root.geometry(
            "1500x900"
        )

        self.root.minsize(
            1200,
            750
        )

        # ----------------------------------------------------
        # 数据
        # ----------------------------------------------------

        self.track_df = safe_read_csv(
            TRACK_CSV
        )

        self.count_df = safe_read_csv(
            COUNT_CSV
        )

        self.dwell_event_df = safe_read_csv(
            DWELL_EVENT_CSV
        )

        self.dwell_summary_df = safe_read_csv(
            DWELL_SUMMARY_CSV
        )

        self.heatmap_df = safe_read_csv(
            HEATMAP_CSV
        )

        # ----------------------------------------------------
        # 计算基础指标
        # ----------------------------------------------------

        self.calculate_statistics()

        # ----------------------------------------------------
        # 创建 UI
        # ----------------------------------------------------

        self.create_ui()


    # ========================================================
    # 数据统计
    # ========================================================

    def calculate_statistics(self):

        # ----------------------------------------------------
        # Global ID
        # ----------------------------------------------------

        if (
            not self.track_df.empty
            and
            "global_id" in self.track_df.columns
        ):

            self.global_ids = (
                self.track_df[
                    "global_id"
                ].nunique()
            )

        else:

            self.global_ids = 0


        # ----------------------------------------------------
        # IN / OUT
        # ----------------------------------------------------

        self.in_count = 0
        self.out_count = 0

        if (
            not self.count_df.empty
            and
            "direction" in self.count_df.columns
        ):

            direction = (
                self.count_df[
                    "direction"
                ]
                .astype(str)
                .str.upper()
            )

            self.in_count = int(
                (direction == "IN").sum()
            )

            self.out_count = int(
                (direction == "OUT").sum()
            )


        self.total_count = (
            self.in_count +
            self.out_count
        )


        # ----------------------------------------------------
        # 区域访问
        # ----------------------------------------------------

        self.visit_count = 0

        self.total_dwell = 0

        self.average_dwell = 0

        self.max_dwell = 0

        self.dwell_people = 0


        if not self.dwell_event_df.empty:

            self.visit_count = len(
                self.dwell_event_df
            )

            if "dwell_time" in self.dwell_event_df.columns:

                self.total_dwell = float(
                    self.dwell_event_df[
                        "dwell_time"
                    ].sum()
                )

                self.average_dwell = float(
                    self.dwell_event_df[
                        "dwell_time"
                    ].mean()
                )

                self.max_dwell = float(
                    self.dwell_event_df[
                        "dwell_time"
                    ].max()
                )

            if "global_id" in self.dwell_event_df.columns:

                self.dwell_people = (
                    self.dwell_event_df[
                        "global_id"
                    ].nunique()
                )


        # ----------------------------------------------------
        # 视频
        # ----------------------------------------------------

        cap = cv2.VideoCapture(
            VIDEO_PATH
        )

        if cap.isOpened():

            self.width = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_WIDTH
                )
            )

            self.height = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_HEIGHT
                )
            )

            self.fps = cap.get(
                cv2.CAP_PROP_FPS
            )

            self.total_frames = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            )

            cap.release()

        else:

            self.width = 0
            self.height = 0
            self.fps = 30
            self.total_frames = 0


        if self.fps <= 0:

            self.fps = 30


        self.duration = (
            self.total_frames /
            self.fps
            if self.fps > 0
            else 0
        )


    # ========================================================
    # 创建 UI
    # ========================================================

    def create_ui(self):

        # ====================================================
        # 标题
        # ====================================================

        title_frame = tk.Frame(
            self.root,
            bg="#202020",
            height=65
        )

        title_frame.pack(
            fill="x"
        )

        title_frame.pack_propagate(
            False
        )

        title = tk.Label(
            title_frame,
            text="人流进出计数与区域热度分析系统",
            font=(
                "Microsoft YaHei",
                22,
                "bold"
            ),
            fg="white",
            bg="#202020"
        )

        title.pack(
            pady=15
        )


        # ====================================================
        # KPI
        # ====================================================

        kpi_frame = tk.Frame(
            self.root,
            bg="#eeeeee"
        )

        kpi_frame.pack(
            fill="x",
            padx=10,
            pady=10
        )


        self.create_kpi(
            kpi_frame,
            "Global ID",
            self.global_ids,
            0
        )

        self.create_kpi(
            kpi_frame,
            "IN 人数",
            self.in_count,
            1
        )

        self.create_kpi(
            kpi_frame,
            "OUT 人数",
            self.out_count,
            2
        )

        self.create_kpi(
            kpi_frame,
            "总进出人数",
            self.total_count,
            3
        )


        # ====================================================
        # 主区域
        # ====================================================

        main_frame = tk.Frame(
            self.root
        )

        main_frame.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=5
        )


        # ====================================================
        # 左侧
        # ====================================================

        left_frame = tk.Frame(
            main_frame
        )

        left_frame.pack(
            side="left",
            fill="both",
            expand=True
        )


        # 视频信息
        video_info = tk.LabelFrame(
            left_frame,
            text="视频信息",
            font=(
                "Microsoft YaHei",
                11,
                "bold"
            )
        )

        video_info.pack(
            fill="x",
            pady=5
        )


        info_text = (
            f"分辨率："
            f"{self.width} × {self.height}    "
            f"FPS：{self.fps:.2f}    "
            f"帧数：{self.total_frames}    "
            f"时长：{self.duration:.2f}s"
        )

        tk.Label(
            video_info,
            text=info_text,
            font=(
                "Microsoft YaHei",
                10
            )
        ).pack(
            pady=8
        )


        # ====================================================
        # 热力图
        # ====================================================

        heat_frame = tk.LabelFrame(
            left_frame,
            text="人员活动热力图",
            font=(
                "Microsoft YaHei",
                11,
                "bold"
            )
        )

        heat_frame.pack(
            fill="both",
            expand=True,
            pady=5
        )


        self.heat_label = tk.Label(
            heat_frame
        )

        self.heat_label.pack(
            fill="both",
            expand=True
        )

        self.load_heatmap()


        # ====================================================
        # 右侧
        # ====================================================

        right_frame = tk.Frame(
            main_frame,
            width=400
        )

        right_frame.pack(
            side="right",
            fill="y",
            padx=(10, 0)
        )

        right_frame.pack_propagate(
            False
        )


        # ====================================================
        # 区域停留
        # ====================================================

        dwell_frame = tk.LabelFrame(
            right_frame,
            text="区域停留统计",
            font=(
                "Microsoft YaHei",
                11,
                "bold"
            )
        )

        dwell_frame.pack(
            fill="x",
            pady=5
        )


        self.create_stat(
            dwell_frame,
            "区域访问人数",
            self.dwell_people
        )

        self.create_stat(
            dwell_frame,
            "区域访问次数",
            self.visit_count
        )

        self.create_stat(
            dwell_frame,
            "平均停留时间",
            f"{self.average_dwell:.2f} s"
        )

        self.create_stat(
            dwell_frame,
            "最大停留时间",
            f"{self.max_dwell:.2f} s"
        )

        self.create_stat(
            dwell_frame,
            "总停留时间",
            f"{self.total_dwell:.2f} s"
        )


        # ====================================================
        # IN / OUT 图
        # ====================================================

        chart_frame = tk.LabelFrame(
            right_frame,
            text="进出统计",
            font=(
                "Microsoft YaHei",
                11,
                "bold"
            )
        )

        chart_frame.pack(
            fill="both",
            expand=True,
            pady=5
        )


        self.create_count_chart(
            chart_frame
        )


        # ====================================================
        # 文件状态
        # ====================================================

        file_frame = tk.LabelFrame(
            right_frame,
            text="数据文件",
            font=(
                "Microsoft YaHei",
                11,
                "bold"
            )
        )

        file_frame.pack(
            fill="x",
            pady=5
        )


        files = [

            (
                "轨迹",
                TRACK_CSV
            ),

            (
                "进出",
                COUNT_CSV
            ),

            (
                "停留",
                DWELL_EVENT_CSV
            ),

            (
                "热力图",
                HEATMAP_CSV
            )
        ]


        for name, path in files:

            exists = os.path.exists(
                path
            )

            state = (
                "✓"
                if exists
                else "✗"
            )

            color = (
                "green"
                if exists
                else "red"
            )

            tk.Label(
                file_frame,
                text=f"{state} {name}",
                fg=color,
                anchor="w"
            ).pack(
                fill="x",
                padx=10,
                pady=2
            )


        # ====================================================
        # 底部
        # ====================================================

        bottom_frame = tk.Frame(
            self.root,
            height=45
        )

        bottom_frame.pack(
            fill="x",
            padx=10,
            pady=5
        )

        bottom_frame.pack_propagate(
            False
        )


        refresh_button = tk.Button(
            bottom_frame,
            text="刷新数据",
            command=self.refresh,
            font=(
                "Microsoft YaHei",
                10
            ),
            width=12
        )

        refresh_button.pack(
            side="right"
        )


    # ========================================================
    # KPI 卡片
    # ========================================================

    def create_kpi(
        self,
        parent,
        title,
        value,
        column
    ):

        frame = tk.Frame(
            parent,
            bg="white",
            relief="solid",
            bd=1
        )

        frame.grid(
            row=0,
            column=column,
            sticky="nsew",
            padx=5
        )

        parent.grid_columnconfigure(
            column,
            weight=1
        )


        tk.Label(
            frame,
            text=title,
            font=(
                "Microsoft YaHei",
                11
            ),
            bg="white"
        ).pack(
            pady=(8, 0)
        )


        tk.Label(
            frame,
            text=str(value),
            font=(
                "Microsoft YaHei",
                24,
                "bold"
            ),
            bg="white"
        ).pack(
            pady=(2, 8)
        )


    # ========================================================
    # 统计项
    # ========================================================

    def create_stat(
        self,
        parent,
        name,
        value
    ):

        frame = tk.Frame(
            parent
        )

        frame.pack(
            fill="x",
            padx=10,
            pady=6
        )


        tk.Label(
            frame,
            text=name,
            font=(
                "Microsoft YaHei",
                10
            ),
            anchor="w"
        ).pack(
            side="left"
        )


        tk.Label(
            frame,
            text=str(value),
            font=(
                "Microsoft YaHei",
                11,
                "bold"
            ),
            anchor="e"
        ).pack(
            side="right"
        )


    # ========================================================
    # 热力图
    # ========================================================

    def load_heatmap(self):

        if not os.path.exists(
            HEATMAP_IMAGE
        ):

            self.heat_label.config(
                text=(
                    "未找到热力图\n\n"
                    "请先运行 heatmap_v7.py"
                ),
                font=(
                    "Microsoft YaHei",
                    14
                )
            )

            return


        try:

            image = cv2.imread(
                HEATMAP_IMAGE
            )

            if image is None:
                raise RuntimeError(
                    "无法读取热力图"
                )

            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            # 根据 Label 大小缩放
            image = cv2.resize(
                image,
                (
                    700,
                    400
                )
            )

            from PIL import Image
            from PIL import ImageTk

            pil_image = Image.fromarray(
                image
            )

            tk_image = ImageTk.PhotoImage(
                pil_image
            )

            self.heat_label.configure(
                image=tk_image
            )

            self.heat_label.image = tk_image

        except Exception as e:

            self.heat_label.config(
                text=f"热力图加载失败：{e}"
            )


    # ========================================================
    # IN / OUT 柱状图
    # ========================================================

    def create_count_chart(
        self,
        parent
    ):

        figure = plt.Figure(
            figsize=(4, 3),
            dpi=90
        )

        ax = figure.add_subplot(
            111
        )


        categories = [
            "IN",
            "OUT"
        ]

        values = [
            self.in_count,
            self.out_count
        ]


        ax.bar(
            categories,
            values
        )

        ax.set_title(
            "人员进出统计"
        )

        ax.set_ylabel(
            "人数"
        )


        ax.grid(
            axis="y",
            alpha=0.25
        )


        figure.tight_layout()


        canvas = FigureCanvasTkAgg(
            figure,
            parent
        )

        canvas.draw()

        canvas.get_tk_widget().pack(
            fill="both",
            expand=True
        )


        self.count_canvas = canvas


    # ========================================================
    # 刷新
    # ========================================================

    def refresh(self):

        self.track_df = safe_read_csv(
            TRACK_CSV
        )

        self.count_df = safe_read_csv(
            COUNT_CSV
        )

        self.dwell_event_df = safe_read_csv(
            DWELL_EVENT_CSV
        )

        self.dwell_summary_df = safe_read_csv(
            DWELL_SUMMARY_CSV
        )

        self.heatmap_df = safe_read_csv(
            HEATMAP_CSV
        )


        self.calculate_statistics()


        messagebox.showinfo(
            "刷新完成",
            "数据已重新读取。"
        )


# ============================================================
# 4. 主程序
# ============================================================

def main():

    root = tk.Tk()

    app = Dashboard(
        root
    )

    root.mainloop()


if __name__ == "__main__":

    main()