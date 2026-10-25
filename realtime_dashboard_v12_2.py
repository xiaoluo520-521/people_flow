import cv2
import os
import time
import json
import queue
import threading
import tkinter as tk
from tkinter import ttk

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk

from ultralytics import YOLO


# ============================================================
# 1. 路径
# ============================================================

PROJECT_DIR = r"E:\people_flow"

VIDEO_PATH = os.path.join(
    PROJECT_DIR,
    "videos",
    "test.mp4"
)

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "yolo11n.pt"
)

CONFIG_PATH = os.path.join(
    PROJECT_DIR,
    "config.json"
)


# ============================================================
# 2. 读取配置
# ============================================================

if os.path.exists(CONFIG_PATH):

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    line_cfg = config.get(
        "counting_line",
        {}
    )

    P1 = tuple(
        line_cfg.get(
            "p1",
            [200, 360]
        )
    )

    P2 = tuple(
        line_cfg.get(
            "p2",
            [1100, 360]
        )
    )

    ROI_POINTS = [
        tuple(p)
        for p in config.get(
            "roi",
            [
                [300, 200],
                [900, 200],
                [1000, 500],
                [250, 500]
            ]
        )
    ]

else:

    P1 = (200, 360)

    P2 = (1100, 360)

    ROI_POINTS = [
        (300, 200),
        (900, 200),
        (1000, 500),
        (250, 500)
    ]


# ============================================================
# 3. Dashboard
# ============================================================

class RealtimeDashboard:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "人流进出计数与区域热度分析 - 实时 Dashboard"
        )

        self.root.geometry(
            "1600x950"
        )

        self.root.minsize(
            1250,
            800
        )

        # ----------------------------------------------------
        # 视频
        # ----------------------------------------------------

        self.cap = cv2.VideoCapture(
            VIDEO_PATH
        )

        if not self.cap.isOpened():

            raise RuntimeError(
                f"视频打开失败：{VIDEO_PATH}"
            )

        self.video_fps = self.cap.get(
            cv2.CAP_PROP_FPS
        )

        if self.video_fps <= 0:
            self.video_fps = 30

        self.total_frames = int(
            self.cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        self.width = int(
            self.cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        self.height = int(
            self.cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )

        )

        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        print("正在加载 YOLO11...")

        self.model = YOLO(
            MODEL_PATH
        )

        print("YOLO11 加载完成")

        # ----------------------------------------------------
        # 播放状态
        # ----------------------------------------------------

        self.running = False

        self.paused = False

        self.stop_flag = False

        self.worker_thread = None

        self.frame_queue = queue.Queue(
            maxsize=1
        )

        # ----------------------------------------------------
        # 实时统计
        # ----------------------------------------------------

        self.current_people = 0

        self.in_count = 0

        self.out_count = 0

        self.track_ids = set()

        # ----------------------------------------------------
        # ROI
        # ----------------------------------------------------

        self.roi_visitors = set()

        self.roi_inside = set()

        self.roi_visits = 0

        # ----------------------------------------------------
        # 计数线
        # ----------------------------------------------------

        self.previous_side = {}

        self.counted_ids = set()

        # ----------------------------------------------------
        # 实时历史数据
        # ----------------------------------------------------

        self.history_time = []

        self.history_people = []

        self.history_in = []

        self.history_out = []

        self.history_roi = []

        self.start_time = time.time()

        # ----------------------------------------------------
        # 热力图
        # ----------------------------------------------------

        self.heatmap = np.zeros(
            (
                self.height,
                self.width
            ),
            dtype=np.float32
        )

        self.heatmap_lock = threading.Lock()

        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        self.fps_counter = 0

        self.fps_start = time.time()

        self.current_fps = 0

        # ----------------------------------------------------
        # UI
        # ----------------------------------------------------

        self.create_ui()

        self.show_first_frame()

        self.root.after(
            30,
            self.update_ui
        )

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.close
        )

    # ========================================================
    # UI
    # ========================================================

    def create_ui(self):

        title = tk.Label(
            self.root,
            text="人流进出计数与区域热度分析",
            font=(
                "Microsoft YaHei",
                22,
                "bold"
            )
        )

        title.pack(
            pady=8
        )

        # ====================================================
        # 上半部分
        # ====================================================

        top_frame = tk.Frame(
            self.root
        )

        top_frame.pack(
            fill=tk.BOTH,
            expand=True,
            padx=10
        )

        # ====================================================
        # 左：视频
        # ====================================================

        video_frame = tk.Frame(
            top_frame
        )

        video_frame.pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True
        )

        video_title = tk.Label(
            video_frame,
            text="实时视频",
            font=(
                "Microsoft YaHei",
                15,
                "bold"
            )
        )

        video_title.pack(
            pady=3
        )

        self.video_label = tk.Label(
            video_frame,
            bg="black"
        )

        self.video_label.pack(
            fill=tk.BOTH,
            expand=True
        )

        # ====================================================
        # 右：KPI
        # ====================================================

        kpi_frame = tk.Frame(
            top_frame,
            width=330
        )

        kpi_frame.pack(
            side=tk.RIGHT,
            fill=tk.Y,
            padx=(12, 0)
        )

        kpi_frame.pack_propagate(
            False
        )

        kpi_title = tk.Label(
            kpi_frame,
            text="实时数据",
            font=(
                "Microsoft YaHei",
                18,
                "bold"
            )
        )

        kpi_title.pack(
            pady=8
        )

        self.kpis = {}

        kpi_list = [
            ("people", "当前人数"),
            ("in", "进入人数"),
            ("out", "离开人数"),
            ("total", "累计进出"),
            ("roi_current", "ROI当前人数"),
            ("roi_visitors", "ROI访问人数"),
            ("roi_visits", "ROI访问次数"),
            ("tracks", "当前Track ID"),
            ("fps", "实时FPS")
        ]

        for key, name in kpi_list:

            frame = tk.Frame(
                kpi_frame,
                bd=1,
                relief=tk.GROOVE
            )

            frame.pack(
                fill=tk.X,
                pady=3
            )

            tk.Label(
                frame,
                text=name,
                font=(
                    "Microsoft YaHei",
                    10
                )
            ).pack(
                side=tk.LEFT,
                padx=8,
                pady=7
            )

            value = tk.Label(
                frame,
                text="0",
                font=(
                    "Microsoft YaHei",
                    13,
                    "bold"
                )
            )

            value.pack(
                side=tk.RIGHT,
                padx=8
            )

            self.kpis[key] = value

        # ====================================================
        # 控制按钮
        # ====================================================

        control_frame = tk.Frame(
            self.root
        )

        control_frame.pack(
            fill=tk.X,
            padx=10,
            pady=5
        )

        tk.Button(
            control_frame,
            text="▶ 播放",
            width=10,
            command=self.play_video
        ).pack(
            side=tk.LEFT,
            padx=4
        )

        tk.Button(
            control_frame,
            text="⏸ 暂停",
            width=10,
            command=self.pause_video
        ).pack(
            side=tk.LEFT,
            padx=4
        )

        tk.Button(
            control_frame,
            text="↻ 重播",
            width=10,
            command=self.restart_video
        ).pack(
            side=tk.LEFT,
            padx=4
        )

        tk.Button(
            control_frame,
            text="■ 停止",
            width=10,
            command=self.stop_video
        ).pack(
            side=tk.LEFT,
            padx=4
        )

        self.status_label = tk.Label(
            control_frame,
            text="状态：已准备",
            font=(
                "Microsoft YaHei",
                10
            )
        )

        self.status_label.pack(
            side=tk.LEFT,
            padx=20
        )

        # ====================================================
        # 进度条
        # ====================================================

        self.progress = ttk.Progressbar(
            self.root,
            orient="horizontal",
            mode="determinate"
        )

        self.progress.pack(
            fill=tk.X,
            padx=15,
            pady=5
        )

        # ====================================================
        # 下半部分：实时曲线
        # ====================================================

        chart_frame = tk.LabelFrame(
            self.root,
            text="实时人流趋势",
            font=(
                "Microsoft YaHei",
                12,
                "bold"
            )
        )

        chart_frame.pack(
            fill=tk.BOTH,
            expand=True,
            padx=10,
            pady=5
        )

        self.fig, self.ax = plt.subplots(
            figsize=(10, 3.0),
            dpi=90
        )

        self.canvas = FigureCanvasTkAgg(
            self.fig,
            master=chart_frame
        )

        self.canvas.get_tk_widget().pack(
            fill=tk.BOTH,
            expand=True
        )

    # ========================================================
    # 首帧
    # ========================================================

    def show_first_frame(self):

        self.cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            0
        )

        ret, frame = self.cap.read()

        if ret:

            frame = self.draw_overlay(
                frame
            )

            self.display_frame(
                frame
            )

        self.cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            0
        )

    # ========================================================
    # 播放
    # ========================================================

    def play_video(self):

        if self.running:

            self.paused = False

            self.status_label.config(
                text="状态：正在播放"
            )

            return

        self.running = True

        self.paused = False

        self.stop_flag = False

        self.status_label.config(
            text="状态：正在播放"
        )

        self.worker_thread = threading.Thread(
            target=self.video_worker,
            daemon=True
        )

        self.worker_thread.start()

    # ========================================================
    # 暂停
    # ========================================================

    def pause_video(self):

        if self.running:

            self.paused = True

            self.status_label.config(
                text="状态：已暂停"
            )

    # ========================================================
    # 停止
    # ========================================================

    def stop_video(self):

        self.running = False

        self.paused = True

        self.stop_flag = True

        self.status_label.config(
            text="状态：已停止"
        )

    # ========================================================
    # 重播
    # ========================================================

    def restart_video(self):

        self.running = False

        self.paused = False

        self.stop_flag = True

        time.sleep(
            0.1
        )

        self.cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            0
        )

        # 清空队列

        while not self.frame_queue.empty():

            try:

                self.frame_queue.get_nowait()

            except queue.Empty:

                break

        # 清空统计

        self.current_people = 0

        self.in_count = 0

        self.out_count = 0

        self.track_ids.clear()

        self.roi_visitors.clear()

        self.roi_inside.clear()

        self.roi_visits = 0

        self.previous_side.clear()

        self.counted_ids.clear()

        # 清空历史

        self.history_time.clear()

        self.history_people.clear()

        self.history_in.clear()

        self.history_out.clear()

        self.history_roi.clear()

        self.start_time = time.time()

        # 清空热力图

        with self.heatmap_lock:

            self.heatmap.fill(0)

        self.show_first_frame()

        self.update_chart()

        self.status_label.config(
            text="状态：已重置"
        )

    # ========================================================
    # 视频线程
    # ========================================================

    def video_worker(self):

        while (
            self.running
            and
            not self.stop_flag
        ):

            if self.paused:

                time.sleep(
                    0.05
                )

                continue

            ret, frame = self.cap.read()

            if not ret:

                self.running = False

                self.root.after(
                    0,
                    lambda:
                    self.status_label.config(
                        text="状态：播放结束"
                    )
                )

                break

            # ------------------------------------------------
            # YOLO
            # ------------------------------------------------

            results = self.model.track(
                frame,
                tracker="bytetrack.yaml",
                persist=True,
                classes=[0],
                conf=0.3,
                verbose=False
            )

            result = results[0]

            ids_now = []

            if (
                result.boxes is not None
                and
                result.boxes.id is not None
            ):

                ids = (
                    result.boxes.id
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                boxes = (
                    result.boxes.xyxy
                    .cpu()
                    .numpy()
                )

                ids_now = ids.tolist()

                self.current_people = len(
                    ids_now
                )

                self.track_ids = set(
                    ids_now
                )

                # ============================================
                # 每个人
                # ============================================

                for track_id, box in zip(
                    ids,
                    boxes
                ):

                    track_id = int(
                        track_id
                    )

                    x1, y1, x2, y2 = box

                    cx = (
                        x1 + x2
                    ) / 2

                    cy = (
                        y1 + y2
                    ) / 2

                    point = (
                        int(cx),
                        int(cy)
                    )

                    # ----------------------------------------
                    # 计数线
                    # ----------------------------------------

                    side = self.get_side(
                        point
                    )

                    if track_id not in self.previous_side:

                        self.previous_side[
                            track_id
                        ] = side

                    else:

                        old_side = self.previous_side[
                            track_id
                        ]

                        # 上 -> 下
                        if (
                            old_side < 0
                            and side > 0
                        ):

                            if (
                                track_id
                                not in self.counted_ids
                            ):

                                self.in_count += 1

                                self.counted_ids.add(
                                    track_id
                                )

                        # 下 -> 上
                        elif (
                            old_side > 0
                            and side < 0
                        ):

                            if (
                                track_id
                                not in self.counted_ids
                            ):

                                self.out_count += 1

                                self.counted_ids.add(
                                    track_id
                                )

                        self.previous_side[
                            track_id
                        ] = side

                    # ----------------------------------------
                    # ROI
                    # ----------------------------------------

                    inside = self.point_in_roi(
                        point
                    )

                    if inside:

                        self.roi_visitors.add(
                            track_id
                        )

                        if (
                            track_id
                            not in self.roi_inside
                        ):

                            self.roi_inside.add(
                                track_id
                            )

                            self.roi_visits += 1

                    else:

                        self.roi_inside.discard(
                            track_id
                        )

                    # ----------------------------------------
                    # 热力图
                    # ----------------------------------------

                    ix = int(
                        max(
                            0,
                            min(
                                self.width - 1,
                                cx
                            )
                        )
                    )

                    iy = int(
                        max(
                            0,
                            min(
                                self.height - 1,
                                cy
                            )
                        )
                    )

                    with self.heatmap_lock:

                        self.heatmap[
                            iy,
                            ix
                        ] += 1

            else:

                self.current_people = 0

            # ------------------------------------------------
            # 绘制
            # ------------------------------------------------

            frame = self.draw_overlay(
                frame,
                result
            )

            # ------------------------------------------------
            # FPS
            # ------------------------------------------------

            self.fps_counter += 1

            now = time.time()

            if (
                now
                -
                self.fps_start
                >= 1
            ):

                self.current_fps = (
                    self.fps_counter
                    /
                    (
                        now
                        -
                        self.fps_start
                    )
                )

                self.fps_counter = 0

                self.fps_start = now

            # ------------------------------------------------
            # 放入队列
            # ------------------------------------------------

            try:

                if self.frame_queue.full():

                    self.frame_queue.get_nowait()

                self.frame_queue.put_nowait(
                    frame
                )

            except queue.Full:

                pass

    # ========================================================
    # UI刷新
    # ========================================================

    def update_ui(self):

        try:

            frame = self.frame_queue.get_nowait()

            self.display_frame(
                frame
            )

        except queue.Empty:

            pass

        # ----------------------------------------------------
        # KPI
        # ----------------------------------------------------

        self.kpis[
            "people"
        ].config(
            text=str(
                self.current_people
            )
        )

        self.kpis[
            "in"
        ].config(
            text=str(
                self.in_count
            )
        )

        self.kpis[
            "out"
        ].config(
            text=str(
                self.out_count
            )
        )

        self.kpis[
            "total"
        ].config(
            text=str(
                self.in_count
                +
                self.out_count
            )
        )

        self.kpis[
            "roi_current"
        ].config(
            text=str(
                len(
                    self.roi_inside
                )
            )
        )

        self.kpis[
            "roi_visitors"
        ].config(
            text=str(
                len(
                    self.roi_visitors
                )
            )
        )

        self.kpis[
            "roi_visits"
        ].config(
            text=str(
                self.roi_visits
            )
        )

        self.kpis[
            "tracks"
        ].config(
            text=str(
                len(
                    self.track_ids
                )
            )
        )

        self.kpis[
            "fps"
        ].config(
            text=f"{self.current_fps:.1f}"
        )

        # ----------------------------------------------------
        # 历史数据
        # ----------------------------------------------------

        elapsed = (
            self.cap.get(
                cv2.CAP_PROP_POS_FRAMES
            )
            /
            self.video_fps
        )

        self.history_time.append(
            elapsed
        )

        self.history_people.append(
            self.current_people
        )

        self.history_in.append(
            self.in_count
        )

        self.history_out.append(
            self.out_count
        )

        self.history_roi.append(
            len(
                self.roi_inside
            )
        )

        # 最多保留最近120秒
        max_points = 3600

        if len(self.history_time) > max_points:

            self.history_time = (
                self.history_time[-max_points:]
            )

            self.history_people = (
                self.history_people[-max_points:]
            )

            self.history_in = (
                self.history_in[-max_points:]
            )

            self.history_out = (
                self.history_out[-max_points:]
            )

            self.history_roi = (
                self.history_roi[-max_points:]
            )

        # ----------------------------------------------------
        # 曲线
        # ----------------------------------------------------

        self.update_chart()

        # ----------------------------------------------------
        # 进度
        # ----------------------------------------------------

        current_frame = int(
            self.cap.get(
                cv2.CAP_PROP_POS_FRAMES
            )
        )

        if self.total_frames > 0:

            self.progress["value"] = (
                current_frame
                /
                self.total_frames
                *
                100
            )

        self.root.after(
            300,
            self.update_ui
        )

    # ========================================================
    # 实时趋势图
    # ========================================================

    def update_chart(self):

        self.ax.clear()

        if len(self.history_time) > 0:

            self.ax.plot(
                self.history_time,
                self.history_people,
                label="当前人数"
            )

            self.ax.plot(
                self.history_time,
                self.history_in,
                label="累计IN"
            )

            self.ax.plot(
                self.history_time,
                self.history_out,
                label="累计OUT"
            )

            self.ax.plot(
                self.history_time,
                self.history_roi,
                label="ROI当前人数"
            )

        self.ax.set_xlabel(
            "时间 / s"
        )

        self.ax.set_ylabel(
            "人数"
        )

        self.ax.set_title(
            "实时人流趋势"
        )

        self.ax.legend(
            loc="upper left"
        )

        self.ax.grid(
            True,
            alpha=0.3
        )

        self.fig.tight_layout()

        self.canvas.draw_idle()

    # ========================================================
    # 视频显示
    # ========================================================

    def display_frame(
        self,
        frame
    ):

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(
            rgb
        )

        max_width = 950
        max_height = 520

        ratio = min(
            max_width / image.width,
            max_height / image.height
        )

        size = (
            int(image.width * ratio),
            int(image.height * ratio)
        )

        image = image.resize(
            size,
            Image.Resampling.LANCZOS
        )

        photo = ImageTk.PhotoImage(
            image
        )

        self.video_label.config(
            image=photo
        )

        self.video_label.image = photo

    # ========================================================
    # 绘制视频
    # ========================================================

    def draw_overlay(
        self,
        frame,
        result=None
    ):

        # ----------------------------------------------------
        # ROI
        # ----------------------------------------------------

        if len(ROI_POINTS) >= 3:

            pts = np.array(
                ROI_POINTS,
                dtype=np.int32
            )

            overlay = frame.copy()

            cv2.fillPoly(
                overlay,
                [pts],
                (255, 0, 0)
            )

            frame = cv2.addWeighted(
                overlay,
                0.12,
                frame,
                0.88,
                0
            )

            cv2.polylines(
                frame,
                [pts],
                True,
                (255, 0, 0),
                3
            )

        # ----------------------------------------------------
        # 计数线
        # ----------------------------------------------------

        cv2.line(
            frame,
            P1,
            P2,
            (0, 255, 255),
            4
        )

        # ----------------------------------------------------
        # 检测框
        # ----------------------------------------------------

        if result is not None:

            if (
                result.boxes is not None
                and
                result.boxes.id is not None
            ):

                ids = (
                    result.boxes.id
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                boxes = (
                    result.boxes.xyxy
                    .cpu()
                    .numpy()
                )

                for track_id, box in zip(
                    ids,
                    boxes
                ):

                    x1, y1, x2, y2 = map(
                        int,
                        box
                    )

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    cx = int(
                        (x1 + x2) / 2
                    )

                    cy = int(
                        (y1 + y2) / 2
                    )

                    cv2.circle(
                        frame,
                        (cx, cy),
                        4,
                        (0, 0, 255),
                        -1
                    )

                    cv2.putText(
                        frame,
                        f"ID {int(track_id)}",
                        (
                            x1,
                            max(
                                20,
                                y1 - 5
                            )
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

        # ----------------------------------------------------
        # 左上角统计
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (10, 10),
            (330, 165),
            (0, 0, 0),
            -1
        )

        texts = [
            f"People: {self.current_people}",
            f"IN: {self.in_count}",
            f"OUT: {self.out_count}",
            f"ROI: {len(self.roi_inside)}",
            f"ROI Visits: {self.roi_visits}"
        ]

        for i, text in enumerate(
            texts
        ):

            cv2.putText(
                frame,
                text,
                (
                    20,
                    40 + i * 25
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

        return frame

    # ========================================================
    # 计数线
    # ========================================================

    def get_side(
        self,
        point
    ):

        x, y = point

        x1, y1 = P1

        x2, y2 = P2

        value = (
            (x2 - x1)
            *
            (y - y1)
            -
            (y2 - y1)
            *
            (x - x1)
        )

        if value > 0:

            return 1

        if value < 0:

            return -1

        return 0

    # ========================================================
    # ROI
    # ========================================================

    def point_in_roi(
        self,
        point
    ):

        if len(ROI_POINTS) < 3:

            return False

        pts = np.array(
            ROI_POINTS,
            dtype=np.int32
        )

        result = cv2.pointPolygonTest(
            pts,
            point,
            False
        )

        return result >= 0

    # ========================================================
    # 关闭
    # ========================================================

    def close(self):

        self.running = False

        self.stop_flag = True

        time.sleep(
            0.1
        )

        if self.cap is not None:

            self.cap.release()

        self.root.destroy()


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":

    print("=" * 65)
    print("人流进出计数与区域热度分析")
    print("V12.2 实时 Dashboard")
    print("=" * 65)

    print(
        f"视频：{VIDEO_PATH}"
    )

    print(
        f"模型：{MODEL_PATH}"
    )

    print(
        f"分辨率："
        f"{1280} × {720}"
    )

    if not os.path.exists(
        VIDEO_PATH
    ):

        print(
            "❌ 视频不存在"
        )

        exit()

    if not os.path.exists(
        MODEL_PATH
    ):

        print(
            "❌ YOLO模型不存在"
        )

        print(
            MODEL_PATH
        )

        exit()

    root = tk.Tk()

    app = RealtimeDashboard(
        root
    )

    root.mainloop()