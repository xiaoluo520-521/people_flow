import os
import json
import time
import threading
import queue
from collections import defaultdict, deque

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk
from ultralytics import YOLO


# ============================================================
# 路径
# ============================================================

PROJECT_DIR = r"E:\people_flow"

CONFIG_FILE = os.path.join(
    PROJECT_DIR,
    "config.json"
)

MODEL_FILE = os.path.join(
    PROJECT_DIR,
    "models",
    "yolo11n.pt"
)


# ============================================================
# 读取配置
# ============================================================

with open(
    CONFIG_FILE,
    "r",
    encoding="utf-8"
) as f:

    config = json.load(f)


VIDEO_PATH = config["video"]

LINE_P1 = tuple(
    config["counting_line"]["p1"]
)

LINE_P2 = tuple(
    config["counting_line"]["p2"]
)

ROI_POINTS = np.array(
    config["roi"],
    dtype=np.int32
)

PARAMS = config.get(
    "parameters",
    {}
)

MIN_TRACK_FRAMES = PARAMS.get(
    "min_track_frames",
    10
)

MIN_MOVEMENT_DISTANCE = PARAMS.get(
    "min_movement_distance",
    10
)


# ============================================================
# 输出
# ============================================================

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "results",
    "realtime_v12_1"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 实时分析类
# ============================================================

class RealtimeProcessor:

    def __init__(self, app):

        self.app = app

        self.running = False
        self.paused = False

        self.cap = None

        self.thread = None

        self.model = None

        # ----------------------------------------------------
        # 统计
        # ----------------------------------------------------

        self.in_count = 0
        self.out_count = 0

        self.counted_ids = set()

        self.current_people = 0

        self.current_roi_people = 0

        self.roi_visitors = set()

        self.roi_visits = 0

        # ----------------------------------------------------
        # Track
        # ----------------------------------------------------

        self.track_history = defaultdict(
            lambda: deque(maxlen=30)
        )

        self.track_frames = defaultdict(
            int
        )

        # ----------------------------------------------------
        # 视频
        # ----------------------------------------------------

        self.current_frame = 0

        self.total_frames = 0

        self.video_fps = 30

        self.current_time = 0

        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        self.processing_fps = 0

        self.fps_counter = 0

        self.fps_start = time.time()

        # ----------------------------------------------------
        # UI队列
        # ----------------------------------------------------

        self.frame_queue = queue.Queue(
            maxsize=1
        )


    # ========================================================
    # 加载模型
    # ========================================================

    def load_model(self):

        if self.model is None:

            print()
            print("正在加载 YOLO11...")

            self.model = YOLO(
                MODEL_FILE
            )

            print("YOLO11 加载完成")


    # ========================================================
    # 打开视频
    # ========================================================

    def open_video(self):

        self.cap = cv2.VideoCapture(
            VIDEO_PATH
        )

        if not self.cap.isOpened():

            raise RuntimeError(
                f"无法打开视频：{VIDEO_PATH}"
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


        width = int(
            self.cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            self.cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )


        print()
        print("=" * 60)
        print("视频信息")
        print("=" * 60)

        print(
            f"视频：{VIDEO_PATH}"
        )

        print(
            f"分辨率：{width} × {height}"
        )

        print(
            f"FPS：{self.video_fps:.2f}"
        )

        print(
            f"帧数：{self.total_frames}"
        )


    # ========================================================
    # 判断点在线的哪一侧
    # ========================================================

    def line_side(self, point):

        x, y = point

        x1, y1 = LINE_P1
        x2, y2 = LINE_P2

        return (
            (x2 - x1) * (y - y1)
            -
            (y2 - y1) * (x - x1)
        )


    # ========================================================
    # 判断是否穿线
    # ========================================================

    def crossed_line(
        self,
        p1,
        p2
    ):

        s1 = self.line_side(
            p1
        )

        s2 = self.line_side(
            p2
        )

        return (
            (s1 < 0 and s2 > 0)
            or
            (s1 > 0 and s2 < 0)
        )


    # ========================================================
    # ROI
    # ========================================================

    def point_in_roi(
        self,
        point
    ):

        return (
            cv2.pointPolygonTest(
                ROI_POINTS,
                point,
                False
            ) >= 0
        )


    # ========================================================
    # 重置
    # ========================================================

    def reset(self):

        self.in_count = 0

        self.out_count = 0

        self.counted_ids.clear()

        self.current_people = 0

        self.current_roi_people = 0

        self.roi_visitors.clear()

        self.roi_visits = 0

        self.track_history.clear()

        self.track_frames.clear()

        self.current_frame = 0

        self.current_time = 0

        self.processing_fps = 0

        self.fps_counter = 0

        self.fps_start = time.time()


    # ========================================================
    # 开始
    # ========================================================

    def start(self):

        if self.running:

            self.paused = False

            return


        self.load_model()

        self.reset()

        self.open_video()

        self.running = True

        self.paused = False


        self.thread = threading.Thread(
            target=self.worker,
            daemon=True
        )

        self.thread.start()


    # ========================================================
    # 暂停
    # ========================================================

    def pause(self):

        self.paused = True


    # ========================================================
    # 停止
    # ========================================================

    def stop(self):

        self.running = False

        self.paused = False


        if self.cap is not None:

            self.cap.release()

            self.cap = None


    # ========================================================
    # 重播
    # ========================================================

    def restart(self):

        self.stop()

        time.sleep(
            0.2
        )

        self.start()


    # ========================================================
    # 处理一帧
    # ========================================================

    def process_frame(
        self,
        frame
    ):

        results = self.model.track(
            frame,
            tracker="bytetrack.yaml",
            persist=True,
            classes=[0],
            conf=0.3,
            verbose=False
        )


        result = results[0]

        current_ids = set()


        if (
            result.boxes is not None
            and result.boxes.id is not None
        ):

            boxes = (
                result.boxes
                .xyxy
                .cpu()
                .numpy()
            )

            ids = (
                result.boxes.id
                .cpu()
                .numpy()
                .astype(int)
            )


            for box, track_id in zip(
                boxes,
                ids
            ):

                x1, y1, x2, y2 = box


                cx = int(
                    (x1 + x2) / 2
                )

                cy = int(
                    (y1 + y2) / 2
                )


                point = (
                    cx,
                    cy
                )


                current_ids.add(
                    track_id
                )


                history = self.track_history[
                    track_id
                ]


                previous = None

                if len(history) > 0:

                    previous = history[-1]


                history.append(
                    point
                )


                self.track_frames[
                    track_id
                ] += 1


                # =================================================
                # 绘制检测框
                # =================================================

                cv2.rectangle(
                    frame,
                    (
                        int(x1),
                        int(y1)
                    ),
                    (
                        int(x2),
                        int(y2)
                    ),
                    (0, 255, 0),
                    2
                )


                cv2.circle(
                    frame,
                    point,
                    4,
                    (0, 0, 255),
                    -1
                )


                cv2.putText(
                    frame,
                    f"ID {track_id}",
                    (
                        int(x1),
                        max(
                            int(y1) - 8,
                            20
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )


                # =================================================
                # 绘制轨迹
                # =================================================

                if len(history) >= 2:

                    for i in range(
                        1,
                        len(history)
                    ):

                        cv2.line(
                            frame,
                            history[i - 1],
                            history[i],
                            (255, 255, 0),
                            2
                        )


                # =================================================
                # 进出判断
                # =================================================

                if (
                    previous is not None
                    and
                    self.track_frames[
                        track_id
                    ] >= MIN_TRACK_FRAMES
                    and
                    track_id not in self.counted_ids
                ):

                    movement = np.linalg.norm(
                        np.array(point)
                        -
                        np.array(previous)
                    )


                    if (
                        movement
                        >=
                        MIN_MOVEMENT_DISTANCE
                    ):

                        if self.crossed_line(
                            previous,
                            point
                        ):

                            before = self.line_side(
                                previous
                            )

                            after = self.line_side(
                                point
                            )


                            if (
                                before < 0
                                and
                                after > 0
                            ):

                                self.in_count += 1

                                self.counted_ids.add(
                                    track_id
                                )


                            elif (
                                before > 0
                                and
                                after < 0
                            ):

                                self.out_count += 1

                                self.counted_ids.add(
                                    track_id
                                )


                # =================================================
                # ROI
                # =================================================

                inside = self.point_in_roi(
                    point
                )


                if inside:

                    self.roi_visitors.add(
                        track_id
                    )


                    if track_id not in self.roi_visitors:

                        self.roi_visits += 1


        # ========================================================
        # 当前人数
        # ========================================================

        self.current_people = len(
            current_ids
        )


        # ========================================================
        # ROI当前人数
        # ========================================================

        roi_current = 0

        for track_id in current_ids:

            history = self.track_history[
                track_id
            ]

            if len(history) == 0:

                continue


            if self.point_in_roi(
                history[-1]
            ):

                roi_current += 1


        self.current_roi_people = roi_current


        # ========================================================
        # 计数线
        # ========================================================

        cv2.line(
            frame,
            LINE_P1,
            LINE_P2,
            (0, 255, 255),
            3
        )


        cv2.putText(
            frame,
            "COUNTING LINE",
            (
                LINE_P1[0],
                max(
                    LINE_P1[1] - 10,
                    20
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )


        # ========================================================
        # ROI
        # ========================================================

        cv2.polylines(
            frame,
            [ROI_POINTS],
            True,
            (255, 0, 255),
            3
        )


        cv2.putText(
            frame,
            "ROI",
            (
                int(ROI_POINTS[0][0]),
                max(
                    int(ROI_POINTS[0][1]) - 10,
                    20
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 255),
            2
        )


        # ========================================================
        # 顶部信息
        # ========================================================

        cv2.rectangle(
            frame,
            (0, 0),
            (430, 105),
            (0, 0, 0),
            -1
        )


        cv2.putText(
            frame,
            f"People: {self.current_people}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"IN: {self.in_count}  OUT: {self.out_count}",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"FPS: {self.processing_fps:.1f}",
            (15, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


        return frame


    # ========================================================
    # Worker
    # ========================================================

    def worker(self):

        while self.running:

            if self.paused:

                time.sleep(
                    0.05
                )

                continue


            ret, frame = self.cap.read()


            if not ret:

                self.running = False

                break


            self.current_time = (
                self.current_frame
                /
                self.video_fps
            )


            frame = self.process_frame(
                frame
            )


            self.current_frame += 1


            # =================================================
            # FPS
            # =================================================

            self.fps_counter += 1

            elapsed = (
                time.time()
                -
                self.fps_start
            )


            if elapsed >= 1:

                self.processing_fps = (
                    self.fps_counter
                    /
                    elapsed
                )

                self.fps_counter = 0

                self.fps_start = time.time()


            # =================================================
            # 队列
            # =================================================

            try:

                if self.frame_queue.full():

                    self.frame_queue.get_nowait()


                self.frame_queue.put_nowait(
                    frame
                )

            except queue.Full:

                pass


            # =================================================
            # 通知UI
            # =================================================

            self.app.root.after(
                0,
                self.app.update_ui
            )


        self.app.root.after(
            0,
            self.app.finished
        )


# ============================================================
# Dashboard
# ============================================================

class Dashboard:

    def __init__(self):

        self.root = tk.Tk()

        self.root.title(
            "人流进出计数与区域热度实时分析系统"
        )

        self.root.geometry(
            "1450x900"
        )

        self.root.minsize(
            1100,
            700
        )


        self.processor = RealtimeProcessor(
            self
        )


        self.photo = None


        self.build_ui()


    # ========================================================
    # 创建UI
    # ========================================================

    def build_ui(self):

        # ====================================================
        # 标题
        # ====================================================

        title = tk.Label(
            self.root,
            text="人流进出计数与区域热度实时分析系统",
            font=(
                "Microsoft YaHei",
                20,
                "bold"
            )
        )

        title.pack(
            pady=10
        )


        # ====================================================
        # 主区域
        # ====================================================

        main = tk.Frame(
            self.root
        )

        main.pack(
            fill=tk.BOTH,
            expand=True,
            padx=10
        )


        # ====================================================
        # 视频区域
        # ====================================================

        video_frame = tk.LabelFrame(
            main,
            text="实时视频",
            font=(
                "Microsoft YaHei",
                12,
                "bold"
            )
        )

        video_frame.pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True,
            padx=(0, 10)
        )


        self.video_label = tk.Label(
            video_frame,
            text="点击“播放”开始实时分析",
            font=(
                "Microsoft YaHei",
                14
            ),
            anchor=tk.CENTER
        )

        self.video_label.pack(
            fill=tk.BOTH,
            expand=True
        )


        # ====================================================
        # 右侧KPI
        # ====================================================

        info_frame = tk.LabelFrame(
            main,
            text="实时数据",
            width=320,
            font=(
                "Microsoft YaHei",
                12,
                "bold"
            )
        )

        info_frame.pack(
            side=tk.RIGHT,
            fill=tk.Y
        )

        info_frame.pack_propagate(
            False
        )


        self.kpi_labels = {}


        self.add_kpi(
            info_frame,
            "当前人数",
            "people"
        )

        self.add_kpi(
            info_frame,
            "IN",
            "in"
        )

        self.add_kpi(
            info_frame,
            "OUT",
            "out"
        )

        self.add_kpi(
            info_frame,
            "累计进出",
            "total"
        )

        self.add_kpi(
            info_frame,
            "ROI当前人数",
            "roi"
        )

        self.add_kpi(
            info_frame,
            "ROI访问次数",
            "roi_visits"
        )

        self.add_kpi(
            info_frame,
            "Global ID",
            "ids"
        )

        self.add_kpi(
            info_frame,
            "处理FPS",
            "fps"
        )


        # ====================================================
        # 控制按钮
        # ====================================================

        control_frame = tk.Frame(
            self.root
        )

        control_frame.pack(
            fill=tk.X,
            pady=8
        )


        tk.Button(
            control_frame,
            text="▶ 播放",
            width=14,
            height=2,
            font=(
                "Microsoft YaHei",
                10,
                "bold"
            ),
            command=self.play
        ).pack(
            side=tk.LEFT,
            padx=8
        )


        tk.Button(
            control_frame,
            text="⏸ 暂停",
            width=14,
            height=2,
            font=(
                "Microsoft YaHei",
                10,
                "bold"
            ),
            command=self.pause
        ).pack(
            side=tk.LEFT,
            padx=8
        )


        tk.Button(
            control_frame,
            text="↻ 重播",
            width=14,
            height=2,
            font=(
                "Microsoft YaHei",
                10,
                "bold"
            ),
            command=self.restart
        ).pack(
            side=tk.LEFT,
            padx=8
        )


        tk.Button(
            control_frame,
            text="⏹ 停止",
            width=14,
            height=2,
            font=(
                "Microsoft YaHei",
                10,
                "bold"
            ),
            command=self.stop
        ).pack(
            side=tk.LEFT,
            padx=8
        )


        # ====================================================
        # 进度条
        # ====================================================

        progress_frame = tk.Frame(
            self.root
        )

        progress_frame.pack(
            fill=tk.X,
            padx=20
        )


        self.progress_var = tk.DoubleVar(
            value=0
        )


        self.progress = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100
        )

        self.progress.pack(
            fill=tk.X
        )


        self.progress_label = tk.Label(
            progress_frame,
            text="0.0%"
        )

        self.progress_label.pack()


        # ====================================================
        # 状态
        # ====================================================

        self.status_label = tk.Label(
            self.root,
            text="状态：等待播放",
            font=(
                "Microsoft YaHei",
                10
            )
        )

        self.status_label.pack(
            pady=5
        )


    # ========================================================
    # KPI
    # ========================================================

    def add_kpi(
        self,
        parent,
        title,
        key
    ):

        frame = tk.Frame(
            parent,
            bd=1,
            relief=tk.SOLID
        )

        frame.pack(
            fill=tk.X,
            padx=10,
            pady=5
        )


        tk.Label(
            frame,
            text=title,
            font=(
                "Microsoft YaHei",
                10
            )
        ).pack(
            pady=2
        )


        value = tk.Label(
            frame,
            text="0",
            font=(
                "Arial",
                22,
                "bold"
            )
        )

        value.pack(
            pady=4
        )


        self.kpi_labels[
            key
        ] = value


    # ========================================================
    # 播放
    # ========================================================

    def play(self):

        self.processor.start()

        self.status_label.config(
            text="状态：正在实时分析..."
        )


    # ========================================================
    # 暂停
    # ========================================================

    def pause(self):

        self.processor.pause()

        self.status_label.config(
            text="状态：已暂停"
        )


    # ========================================================
    # 重播
    # ========================================================

    def restart(self):

        self.processor.restart()

        self.status_label.config(
            text="状态：重新播放..."
        )


    # ========================================================
    # 停止
    # ========================================================

    def stop(self):

        self.processor.stop()

        self.status_label.config(
            text="状态：已停止"
        )


    # ========================================================
    # 更新UI
    # ========================================================

    def update_ui(self):

        # ----------------------------------------------------
        # 视频
        # ----------------------------------------------------

        try:

            frame = (
                self.processor
                .frame_queue
                .get_nowait()
            )

        except queue.Empty:

            frame = None


        if frame is not None:

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )


            image = Image.fromarray(
                frame
            )


            # ------------------------------------------------
            # 根据视频区域自动缩放
            # ------------------------------------------------

            max_width = 850
            max_height = 650


            image.thumbnail(
                (
                    max_width,
                    max_height
                ),
                Image.Resampling.LANCZOS
            )


            self.photo = ImageTk.PhotoImage(
                image
            )


            self.video_label.config(
                image=self.photo,
                text=""
            )


        # ----------------------------------------------------
        # KPI
        # ----------------------------------------------------

        p = self.processor


        self.kpi_labels[
            "people"
        ].config(
            text=str(
                p.current_people
            )
        )


        self.kpi_labels[
            "in"
        ].config(
            text=str(
                p.in_count
            )
        )


        self.kpi_labels[
            "out"
        ].config(
            text=str(
                p.out_count
            )
        )


        self.kpi_labels[
            "total"
        ].config(
            text=str(
                p.in_count
                +
                p.out_count
            )
        )


        self.kpi_labels[
            "roi"
        ].config(
            text=str(
                p.current_roi_people
            )
        )


        self.kpi_labels[
            "roi_visits"
        ].config(
            text=str(
                p.roi_visits
            )
        )


        self.kpi_labels[
            "ids"
        ].config(
            text=str(
                len(
                    p.track_frames
                )
            )
        )


        self.kpi_labels[
            "fps"
        ].config(
            text=f"{p.processing_fps:.1f}"
        )


        # ----------------------------------------------------
        # 进度
        # ----------------------------------------------------

        if p.total_frames > 0:

            progress = (
                p.current_frame
                /
                p.total_frames
                *
                100
            )


            self.progress_var.set(
                progress
            )


            total_time = (
                p.total_frames
                /
                p.video_fps
            )


            self.progress_label.config(
                text=(
                    f"{progress:.1f}%    "
                    f"{p.current_time:.1f}s / "
                    f"{total_time:.1f}s"
                )
            )


    # ========================================================
    # 视频结束
    # ========================================================

    def finished(self):

        self.status_label.config(
            text="状态：视频播放完成"
        )


    # ========================================================
    # 关闭
    # ========================================================

    def close(self):

        self.processor.stop()

        self.root.destroy()


    # ========================================================
    # 主循环
    # ========================================================

    def run(self):

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.close
        )

        self.root.mainloop()


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("人流进出计数与区域热度实时分析系统 V12.1")
    print("=" * 70)

    print()
    print(
        f"视频：{VIDEO_PATH}"
    )

    print(
        f"模型：{MODEL_FILE}"
    )

    print(
        f"计数线：{LINE_P1} -> {LINE_P2}"
    )

    print(
        f"ROI：{ROI_POINTS.tolist()}"
    )

    app = Dashboard()

    app.run()