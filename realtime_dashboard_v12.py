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
# 项目路径
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

if not os.path.exists(CONFIG_FILE):
    raise FileNotFoundError(
        f"找不到配置文件：{CONFIG_FILE}"
    )

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

MAX_FRAME_GAP = PARAMS.get(
    "max_frame_gap",
    10
)

HEATMAP_BLUR_KERNEL = PARAMS.get(
    "heatmap_blur_kernel",
    51
)


# ============================================================
# 输出目录
# ============================================================

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "results",
    "realtime_v12"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 颜色
# ============================================================

LINE_COLOR = (0, 255, 255)
ROI_COLOR = (255, 0, 255)

PERSON_COLOR = (0, 255, 0)
CURRENT_POINT_COLOR = (0, 0, 255)

TEXT_COLOR = (255, 255, 255)


# ============================================================
# 工具函数
# ============================================================

def side_of_line(point, p1, p2):
    """
    判断点在线段的哪一侧。

    返回：
        > 0
        < 0
        = 0
    """

    x, y = point

    x1, y1 = p1
    x2, y2 = p2

    value = (
        (x2 - x1) * (y - y1)
        - (y2 - y1) * (x - x1)
    )

    return value


def point_line_distance(point, p1, p2):
    """
    点到无限直线距离。
    """

    x, y = point

    x1, y1 = p1
    x2, y2 = p2

    numerator = abs(
        (y2 - y1) * x
        - (x2 - x1) * y
        + x2 * y1
        - y2 * x1
    )

    denominator = np.sqrt(
        (y2 - y1) ** 2
        + (x2 - x1) ** 2
    )

    if denominator == 0:
        return 999999

    return numerator / denominator


def line_segment_intersection(
    a,
    b,
    p1,
    p2
):
    """
    判断轨迹线段 AB 是否穿过计数线 P1P2。
    """

    s1 = side_of_line(
        a,
        p1,
        p2
    )

    s2 = side_of_line(
        b,
        p1,
        p2
    )

    return (
        (s1 > 0 and s2 < 0)
        or
        (s1 < 0 and s2 > 0)
    )


# ============================================================
# 实时分析线程
# ============================================================

class RealtimeProcessor:

    def __init__(self, app):

        self.app = app

        self.running = False
        self.paused = False

        self.thread = None

        self.cap = None

        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("正在加载 YOLO11...")
        print("=" * 70)

        self.model = YOLO(
            MODEL_FILE
        )

        print("YOLO11 加载完成")


        # ----------------------------------------------------
        # 统计
        # ----------------------------------------------------

        self.in_count = 0
        self.out_count = 0

        self.counted_ids = set()


        # ----------------------------------------------------
        # 轨迹
        # ----------------------------------------------------

        self.track_history = defaultdict(
            lambda: deque(
                maxlen=30
            )
        )

        self.first_seen = {}

        self.last_seen = {}

        self.track_frames = defaultdict(
            int
        )


        # ----------------------------------------------------
        # ROI
        # ----------------------------------------------------

        self.roi_visitors = set()

        self.roi_visits = 0

        self.roi_inside = set()

        self.roi_enter_time = {}


        # ----------------------------------------------------
        # 热力图
        # ----------------------------------------------------

        self.heatmap = None


        # ----------------------------------------------------
        # 时间趋势
        # ----------------------------------------------------

        self.trend_times = []

        self.trend_in = []

        self.trend_out = []

        self.last_trend_second = -1


        # ----------------------------------------------------
        # 当前状态
        # ----------------------------------------------------

        self.current_people = 0

        self.current_roi_people = 0

        self.current_frame = 0

        self.current_time = 0

        self.video_fps = 30

        self.total_frames = 0

        self.processing_fps = 0


        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        self.fps_start_time = time.time()

        self.fps_frame_count = 0


        # ----------------------------------------------------
        # 结果视频
        # ----------------------------------------------------

        self.writer = None


    # ========================================================
    # 初始化视频
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


        output_video = os.path.join(
            OUTPUT_DIR,
            "realtime_analysis_v12.mp4"
        )


        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )


        self.writer = cv2.VideoWriter(
            output_video,
            fourcc,
            self.video_fps,
            (width, height)
        )


        # ----------------------------------------------------
        # 热力图
        # ----------------------------------------------------

        self.heatmap = np.zeros(
            (
                height,
                width
            ),
            dtype=np.float32
        )


        print()
        print("视频信息")
        print("-" * 50)

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
            f"总帧数：{self.total_frames}"
        )


    # ========================================================
    # 开始处理
    # ========================================================

    def start(self):

        if self.running:

            return


        self.running = True

        self.paused = False


        self.open_video()


        self.thread = threading.Thread(
            target=self.process_video,
            daemon=True
        )

        self.thread.start()


    # ========================================================
    # 暂停
    # ========================================================

    def pause(self):

        self.paused = True


    # ========================================================
    # 继续
    # ========================================================

    def resume(self):

        self.paused = False


    # ========================================================
    # 重播
    # ========================================================

    def restart(self):

        self.stop()

        time.sleep(
            0.2
        )

        self.reset_statistics()

        self.start()


    # ========================================================
    # 停止
    # ========================================================

    def stop(self):

        self.running = False

        self.paused = False

        if self.cap is not None:

            self.cap.release()

            self.cap = None


        if self.writer is not None:

            self.writer.release()

            self.writer = None


    # ========================================================
    # 清空统计
    # ========================================================

    def reset_statistics(self):

        self.in_count = 0
        self.out_count = 0

        self.counted_ids.clear()

        self.track_history.clear()

        self.first_seen.clear()

        self.last_seen.clear()

        self.track_frames.clear()

        self.roi_visitors.clear()

        self.roi_visits = 0

        self.roi_inside.clear()

        self.roi_enter_time.clear()

        self.heatmap = None

        self.trend_times.clear()

        self.trend_in.clear()

        self.trend_out.clear()

        self.last_trend_second = -1

        self.current_people = 0

        self.current_roi_people = 0

        self.current_frame = 0

        self.current_time = 0

        self.processing_fps = 0


    # ========================================================
    # ROI判断
    # ========================================================

    def in_roi(self, point):

        result = cv2.pointPolygonTest(
            ROI_POINTS,
            point,
            False
        )

        return result >= 0


    # ========================================================
    # 处理单帧
    # ========================================================

    def process_frame(
        self,
        frame
    ):

        # ----------------------------------------------------
        # YOLO + ByteTrack
        # ----------------------------------------------------

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

            boxes = result.boxes.xyxy.cpu().numpy()

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


                current_ids.add(
                    track_id
                )


                # ------------------------------------------------
                # Track历史
                # ------------------------------------------------

                history = self.track_history[
                    track_id
                ]

                previous_point = (
                    history[-1]
                    if len(history) > 0
                    else None
                )


                history.append(
                    (cx, cy)
                )


                self.track_frames[
                    track_id
                ] += 1


                if track_id not in self.first_seen:

                    self.first_seen[
                        track_id
                    ] = self.current_frame


                self.last_seen[
                    track_id
                ] = self.current_frame


                # ------------------------------------------------
                # 绘制检测框
                # ------------------------------------------------

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
                    PERSON_COLOR,
                    2
                )


                cv2.circle(
                    frame,
                    (
                        cx,
                        cy
                    ),
                    4,
                    CURRENT_POINT_COLOR,
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
                    TEXT_COLOR,
                    2
                )


                # ------------------------------------------------
                # 画轨迹
                # ------------------------------------------------

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


                # ------------------------------------------------
                # 进出计数
                # ------------------------------------------------

                if (
                    previous_point is not None
                    and self.track_frames[
                        track_id
                    ] >= MIN_TRACK_FRAMES
                    and track_id not in self.counted_ids
                ):

                    movement = np.linalg.norm(
                        np.array(
                            [cx, cy]
                        )
                        -
                        np.array(
                            previous_point
                        )
                    )


                    if movement >= MIN_MOVEMENT_DISTANCE:

                        crossed = line_segment_intersection(
                            previous_point,
                            (cx, cy),
                            LINE_P1,
                            LINE_P2
                        )


                        if crossed:

                            before_side = side_of_line(
                                previous_point,
                                LINE_P1,
                                LINE_P2
                            )

                            after_side = side_of_line(
                                (cx, cy),
                                LINE_P1,
                                LINE_P2
                            )


                            # ------------------------------------
                            # 约定：
                            # 上侧 -> 下侧 = IN
                            # 下侧 -> 上侧 = OUT
                            # ------------------------------------

                            if (
                                before_side < 0
                                and after_side > 0
                            ):

                                self.in_count += 1

                                self.counted_ids.add(
                                    track_id
                                )


                            elif (
                                before_side > 0
                                and after_side < 0
                            ):

                                self.out_count += 1

                                self.counted_ids.add(
                                    track_id
                                )


                # ------------------------------------------------
                # ROI
                # ------------------------------------------------

                inside = self.in_roi(
                    (cx, cy)
                )


                if inside:

                    self.roi_visitors.add(
                        track_id
                    )


                    if track_id not in self.roi_inside:

                        self.roi_inside.add(
                            track_id
                        )

                        self.roi_visits += 1

                        self.roi_enter_time[
                            track_id
                        ] = self.current_time


                else:

                    if track_id in self.roi_inside:

                        self.roi_inside.remove(
                            track_id
                        )


                # ------------------------------------------------
                # 热力图
                # ------------------------------------------------

                if (
                    self.heatmap is not None
                    and 0 <= cy < self.heatmap.shape[0]
                    and 0 <= cx < self.heatmap.shape[1]
                ):

                    self.heatmap[
                        cy,
                        cx
                    ] += 1


        # ========================================================
        # 当前人数
        # ========================================================

        self.current_people = len(
            current_ids
        )


        self.current_roi_people = len(
            self.roi_inside
            & current_ids
        )


        # ========================================================
        # 绘制计数线
        # ========================================================

        cv2.line(
            frame,
            LINE_P1,
            LINE_P2,
            LINE_COLOR,
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
            LINE_COLOR,
            2
        )


        # ========================================================
        # 绘制ROI
        # ========================================================

        cv2.polylines(
            frame,
            [
                ROI_POINTS
            ],
            True,
            ROI_COLOR,
            3
        )


        cv2.putText(
            frame,
            "ROI",
            (
                int(ROI_POINTS[0][0]),
                int(ROI_POINTS[0][1]) - 10
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            ROI_COLOR,
            2
        )


        # ========================================================
        # 更新时间趋势
        # ========================================================

        second = int(
            self.current_time
        )


        if second != self.last_trend_second:

            self.last_trend_second = second

            self.trend_times.append(
                second
            )

            self.trend_in.append(
                self.in_count
            )

            self.trend_out.append(
                self.out_count
            )


        # ========================================================
        # FPS
        # ========================================================

        self.fps_frame_count += 1

        elapsed = (
            time.time()
            -
            self.fps_start_time
        )


        if elapsed >= 1.0:

            self.processing_fps = (
                self.fps_frame_count
                /
                elapsed
            )

            self.fps_frame_count = 0

            self.fps_start_time = time.time()


        # ========================================================
        # 视频信息
        # ========================================================

        progress = 0

        if self.total_frames > 0:

            progress = (
                self.current_frame
                /
                self.total_frames
                *
                100
            )


        # ========================================================
        # 顶部实时信息
        # ========================================================

        cv2.rectangle(
            frame,
            (0, 0),
            (520, 100),
            (0, 0, 0),
            -1
        )


        cv2.putText(
            frame,
            f"People: {self.current_people}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            TEXT_COLOR,
            2
        )


        cv2.putText(
            frame,
            f"IN: {self.in_count}  OUT: {self.out_count}",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            TEXT_COLOR,
            2
        )


        cv2.putText(
            frame,
            f"FPS: {self.processing_fps:.1f}",
            (15, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            TEXT_COLOR,
            2
        )


        # ========================================================
        # 保存实时结果视频
        # ========================================================

        if self.writer is not None:

            self.writer.write(
                frame
            )


        return frame


    # ========================================================
    # 视频处理线程
    # ========================================================

    def process_video(self):

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


            processed = self.process_frame(
                frame
            )


            self.current_frame += 1


            # ------------------------------------------------
            # 更新UI
            # ------------------------------------------------

            self.app.frame_queue.put(
                processed
            )


            self.app.root.after(
                1,
                self.app.update_dashboard
            )


        # ========================================================
        # 结束
        # ========================================================

        if self.writer is not None:

            self.writer.release()

            self.writer = None


        if self.cap is not None:

            self.cap.release()

            self.cap = None


        self.app.root.after(
            0,
            self.app.video_finished
        )


# ============================================================
# Dashboard
# ============================================================

class RealtimeDashboard:

    def __init__(self):

        self.root = tk.Tk()

        self.root.title(
            "人流进出计数与区域热度实时分析系统"
        )

        self.root.geometry(
            "1500x900"
        )


        self.processor = RealtimeProcessor(
            self
        )


        self.frame_queue = queue.Queue(
            maxsize=2
        )


        self.current_image = None


        self.build_ui()


    # ========================================================
    # UI
    # ========================================================

    def build_ui(self):

        # ----------------------------------------------------
        # 标题
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # 主区域
        # ----------------------------------------------------

        main = tk.Frame(
            self.root
        )

        main.pack(
            fill=tk.BOTH,
            expand=True,
            padx=10,
            pady=5
        )


        # ----------------------------------------------------
        # 左侧视频
        # ----------------------------------------------------

        left = tk.Frame(
            main,
            bd=2,
            relief=tk.GROOVE
        )

        left.pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True,
            padx=5
        )


        video_title = tk.Label(
            left,
            text="实时视频",
            font=(
                "Microsoft YaHei",
                14,
                "bold"
            )
        )

        video_title.pack(
            pady=5
        )


        self.video_label = tk.Label(
            left
        )

        self.video_label.pack(
            fill=tk.BOTH,
            expand=True
        )


        # ----------------------------------------------------
        # 右侧数据
        # ----------------------------------------------------

        right = tk.Frame(
            main,
            width=350
        )

        right.pack(
            side=tk.RIGHT,
            fill=tk.Y,
            padx=5
        )


        self.create_kpi(
            right,
            "当前人数",
            "current_people"
        )

        self.create_kpi(
            right,
            "IN",
            "in_count"
        )

        self.create_kpi(
            right,
            "OUT",
            "out_count"
        )

        self.create_kpi(
            right,
            "累计进出",
            "total_count"
        )

        self.create_kpi(
            right,
            "ROI当前人数",
            "roi_current"
        )

        self.create_kpi(
            right,
            "ROI访问次数",
            "roi_visits"
        )

        self.create_kpi(
            right,
            "Global ID",
            "global_ids"
        )

        self.create_kpi(
            right,
            "处理FPS",
            "fps"
        )


        # ----------------------------------------------------
        # 视频进度
        # ----------------------------------------------------

        progress_frame = tk.Frame(
            self.root
        )

        progress_frame.pack(
            fill=tk.X,
            padx=20,
            pady=5
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
            text="0%"
        )

        self.progress_label.pack()


        # ----------------------------------------------------
        # 控制按钮
        # ----------------------------------------------------

        controls = tk.Frame(
            self.root
        )

        controls.pack(
            pady=10
        )


        tk.Button(
            controls,
            text="▶ 播放",
            width=12,
            command=self.play
        ).pack(
            side=tk.LEFT,
            padx=5
        )


        tk.Button(
            controls,
            text="⏸ 暂停",
            width=12,
            command=self.pause
        ).pack(
            side=tk.LEFT,
            padx=5
        )


        tk.Button(
            controls,
            text="↻ 重播",
            width=12,
            command=self.restart
        ).pack(
            side=tk.LEFT,
            padx=5
        )


        tk.Button(
            controls,
            text="⏹ 停止",
            width=12,
            command=self.stop
        ).pack(
            side=tk.LEFT,
            padx=5
        )


        # ----------------------------------------------------
        # 状态
        # ----------------------------------------------------

        self.status_label = tk.Label(
            self.root,
            text="等待开始...",
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

    def create_kpi(
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
            pady=4
        )


        tk.Label(
            frame,
            text=title,
            font=(
                "Microsoft YaHei",
                11
            )
        ).pack(
            pady=2
        )


        label = tk.Label(
            frame,
            text="0",
            font=(
                "Arial",
                20,
                "bold"
            )
        )

        label.pack(
            pady=4
        )


        setattr(
            self,
            f"{key}_label",
            label
        )


    # ========================================================
    # 播放
    # ========================================================

    def play(self):

        if not self.processor.running:

            self.processor.start()

            self.status_label.config(
                text="正在实时分析..."
            )

        else:

            self.processor.resume()

            self.status_label.config(
                text="正在播放..."
            )


    # ========================================================
    # 暂停
    # ========================================================

    def pause(self):

        self.processor.pause()

        self.status_label.config(
            text="已暂停"
        )


    # ========================================================
    # 重播
    # ========================================================

    def restart(self):

        self.processor.restart()

        self.status_label.config(
            text="重新开始实时分析..."
        )


    # ========================================================
    # 停止
    # ========================================================

    def stop(self):

        self.processor.stop()

        self.status_label.config(
            text="已停止"
        )


    # ========================================================
    # 更新视频
    # ========================================================

    def update_dashboard(self):

        # ----------------------------------------------------
        # 获取最新视频帧
        # ----------------------------------------------------

        latest_frame = None


        while not self.frame_queue.empty():

            try:

                latest_frame = (
                    self.frame_queue.get_nowait()
                )

            except queue.Empty:

                break


        if latest_frame is not None:

            frame = latest_frame.copy()


            # ------------------------------------------------
            # OpenCV BGR → RGB
            # ------------------------------------------------

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )


            # ------------------------------------------------
            # 缩放
            # ------------------------------------------------

            image = Image.fromarray(
                frame
            )


            image.thumbnail(
                (850, 650)
            )


            photo = ImageTk.PhotoImage(
                image
            )


            self.current_image = photo


            self.video_label.config(
                image=photo
            )


        # ----------------------------------------------------
        # KPI
        # ----------------------------------------------------

        p = self.processor


        self.current_people_label.config(
            text=str(
                p.current_people
            )
        )


        self.in_count_label.config(
            text=str(
                p.in_count
            )
        )


        self.out_count_label.config(
            text=str(
                p.out_count
            )
        )


        self.total_count_label.config(
            text=str(
                p.in_count
                +
                p.out_count
            )
        )


        self.roi_current_label.config(
            text=str(
                p.current_roi_people
            )
        )


        self.roi_visits_label.config(
            text=str(
                p.roi_visits
            )
        )


        self.global_ids_label.config(
            text=str(
                len(
                    p.first_seen
                )
            )
        )


        self.fps_label.config(
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


            self.progress_label.config(
                text=(
                    f"{progress:.1f}%   "
                    f"{p.current_time:.1f}s / "
                    f"{p.total_frames / p.video_fps:.1f}s"
                )
            )


    # ========================================================
    # 视频结束
    # ========================================================

    def video_finished(self):

        self.status_label.config(
            text="视频播放完成，实时分析结束"
        )


    # ========================================================
    # 运行
    # ========================================================

    def run(self):

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.on_close
        )

        self.root.mainloop()


    # ========================================================
    # 关闭
    # ========================================================

    def on_close(self):

        self.processor.stop()

        self.root.destroy()


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("人流进出计数与区域热度实时分析系统 V12")
    print("=" * 70)

    print()
    print("配置文件：")
    print(CONFIG_FILE)

    print()
    print("视频：")
    print(VIDEO_PATH)

    print()
    print("计数线：")
    print(
        f"{LINE_P1} -> {LINE_P2}"
    )

    print()
    print("ROI：")
    print(
        ROI_POINTS.tolist()
    )

    app = RealtimeDashboard()

    app.run()