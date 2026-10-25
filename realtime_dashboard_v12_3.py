import os
import cv2
import json
import time
import threading
import queue
import numpy as np
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

import matplotlib
matplotlib.use("TkAgg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from ultralytics import YOLO


# ============================================================
# 1. 基础配置
# ============================================================

BASE_DIR = r"E:\people_flow"

VIDEO_PATH = os.path.join(BASE_DIR, "videos", "test.mp4")
MODEL_PATH = os.path.join(BASE_DIR, "yolo11n.pt")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

DEVICE = "cuda:0"

# YOLO参数
CONF = 0.3
CLASSES = [0]

# UI刷新频率
VIDEO_UI_INTERVAL = 30       # ms，大约33 FPS
KPI_UI_INTERVAL = 100        # ms
CHART_UI_INTERVAL = 1000     # ms
HEATMAP_UI_INTERVAL = 1000   # ms

# 历史趋势最多保存多少个点
MAX_HISTORY = 300

# 热力图
HEATMAP_BLUR_KERNEL = 51


# ============================================================
# 2. 读取配置
# ============================================================

if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"找不到配置文件：{CONFIG_PATH}")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

LINE_P1 = tuple(CONFIG["counting_line"]["p1"])
LINE_P2 = tuple(CONFIG["counting_line"]["p2"])

ROI_POINTS = [
    tuple(p) for p in CONFIG["roi"]
]

PARAMETERS = CONFIG.get("parameters", {})

MIN_TRACK_FRAMES = PARAMETERS.get("min_track_frames", 10)
MIN_MOVEMENT_DISTANCE = PARAMETERS.get("min_movement_distance", 10)
MAX_FRAME_GAP = PARAMETERS.get("max_frame_gap", 10)


# ============================================================
# 3. 工具函数
# ============================================================

def cross_product(a, b, c):
    """
    计算 AB × AC
    判断点c在线段ab哪一侧
    """
    return (
        (b[0] - a[0]) * (c[1] - a[1])
        - (b[1] - a[1]) * (c[0] - a[0])
    )


def point_side(point):
    """
    根据计数线判断点在线的哪一侧。

    <0 ：一侧
    >0 ：另一侧
    """
    return cross_product(
        LINE_P1,
        LINE_P2,
        point
    )


def point_in_polygon(point, polygon):
    """
    判断点是否在ROI区域内
    """
    polygon_np = np.array(
        polygon,
        dtype=np.int32
    )

    return cv2.pointPolygonTest(
        polygon_np,
        point,
        False
    ) >= 0


def distance(p1, p2):
    return np.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


# ============================================================
# 4. 主Dashboard
# ============================================================

class RealtimeDashboard:

    def __init__(self, root):

        self.root = root
        self.root.title("人流进出计数与区域热度分析 - V12.3")
        self.root.geometry("1500x950")
        self.root.minsize(1200, 800)

        # ----------------------------------------------------
        # 视频
        # ----------------------------------------------------

        self.cap = None

        self.video_width = 1280
        self.video_height = 720

        self.total_frames = 0
        self.fps_video = 30

        self.current_frame_index = 0

        # ----------------------------------------------------
        # 线程控制
        # ----------------------------------------------------

        self.running = False
        self.paused = False
        self.stop_requested = False

        self.worker_thread = None

        # 最新视频帧
        self.frame_lock = threading.Lock()
        self.latest_frame = None

        # 最新统计
        self.data_lock = threading.Lock()

        self.current_people = 0
        self.current_ids = []

        self.in_count = 0
        self.out_count = 0

        self.roi_current = 0
        self.roi_visitors = set()
        self.roi_inside = set()
        self.roi_visits = 0

        # ----------------------------------------------------
        # Track状态
        # ----------------------------------------------------

        self.track_history = {}

        self.track_first_side = {}
        self.track_last_side = {}

        self.track_frame_count = {}

        # ----------------------------------------------------
        # 趋势数据
        # ----------------------------------------------------

        self.history_time = []
        self.history_people = []
        self.history_in = []
        self.history_out = []
        self.history_roi = []

        # ----------------------------------------------------
        # 热力图
        # ----------------------------------------------------

        self.heatmap = np.zeros(
            (self.video_height, self.video_width),
            dtype=np.float32
        )

        self.heatmap_lock = threading.Lock()

        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        self.process_fps = 0.0

        self.last_fps_time = time.time()
        self.fps_counter = 0

        # ----------------------------------------------------
        # UI
        # ----------------------------------------------------

        self.create_ui()

        # ----------------------------------------------------
        # 初始化视频
        # ----------------------------------------------------

        self.open_video()

        # ----------------------------------------------------
        # 加载模型
        # ----------------------------------------------------

        print("=" * 60)
        print("正在加载 YOLO11n...")
        print("设备：", DEVICE)
        print("=" * 60)

        self.model = YOLO(MODEL_PATH)

        self.model.to(DEVICE)

        print("YOLO设备：",
              next(self.model.model.parameters()).device)

        print("模型加载完成")

        # ----------------------------------------------------
        # 定时刷新
        # ----------------------------------------------------

        self.root.after(
            VIDEO_UI_INTERVAL,
            self.update_video_ui
        )

        self.root.after(
            KPI_UI_INTERVAL,
            self.update_kpi
        )

        self.root.after(
            CHART_UI_INTERVAL,
            self.update_chart
        )

        self.root.after(
            HEATMAP_UI_INTERVAL,
            self.update_heatmap
        )

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.on_close
        )


    # ========================================================
    # 创建UI
    # ========================================================

    def create_ui(self):

        # ====================================================
        # 标题
        # ====================================================

        title = tk.Label(
            self.root,
            text="人流进出计数与区域热度分析",
            font=("Microsoft YaHei", 22, "bold")
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

        # ----------------------------------------------------
        # 视频区域
        # ----------------------------------------------------

        video_frame = tk.LabelFrame(
            top_frame,
            text="实时视频",
            font=("Microsoft YaHei", 12)
        )

        video_frame.pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True,
            padx=(0, 5)
        )

        self.video_label = tk.Label(
            video_frame,
            bg="black"
        )

        self.video_label.pack(
            fill=tk.BOTH,
            expand=True
        )

        # ----------------------------------------------------
        # KPI区域
        # ----------------------------------------------------

        right_frame = tk.Frame(
            top_frame,
            width=330
        )

        right_frame.pack(
            side=tk.RIGHT,
            fill=tk.Y
        )

        right_frame.pack_propagate(False)

        kpi_title = tk.Label(
            right_frame,
            text="实时数据",
            font=("Microsoft YaHei", 16, "bold")
        )

        kpi_title.pack(
            pady=10
        )

        self.kpi_current = self.create_kpi(
            right_frame,
            "当前人数"
        )

        self.kpi_in = self.create_kpi(
            right_frame,
            "进入人数"
        )

        self.kpi_out = self.create_kpi(
            right_frame,
            "离开人数"
        )

        self.kpi_total = self.create_kpi(
            right_frame,
            "累计进出"
        )

        self.kpi_roi_current = self.create_kpi(
            right_frame,
            "ROI当前人数"
        )

        self.kpi_roi_visitors = self.create_kpi(
            right_frame,
            "ROI访问人数"
        )

        self.kpi_roi_visits = self.create_kpi(
            right_frame,
            "ROI访问次数"
        )

        self.kpi_fps = self.create_kpi(
            right_frame,
            "处理FPS"
        )

        self.kpi_ids = tk.Label(
            right_frame,
            text="Track IDs：-",
            font=("Microsoft YaHei", 10),
            anchor="w",
            justify=tk.LEFT,
            wraplength=300
        )

        self.kpi_ids.pack(
            fill=tk.X,
            padx=10,
            pady=5
        )

        # ====================================================
        # 控制区域
        # ====================================================

        control_frame = tk.Frame(
            self.root
        )

        control_frame.pack(
            fill=tk.X,
            pady=8
        )

        self.play_button = ttk.Button(
            control_frame,
            text="▶ 播放",
            command=self.start_play
        )

        self.play_button.pack(
            side=tk.LEFT,
            padx=5
        )

        self.pause_button = ttk.Button(
            control_frame,
            text="⏸ 暂停",
            command=self.pause_play
        )

        self.pause_button.pack(
            side=tk.LEFT,
            padx=5
        )

        self.restart_button = ttk.Button(
            control_frame,
            text="↻ 重播",
            command=self.restart_video
        )

        self.restart_button.pack(
            side=tk.LEFT,
            padx=5
        )

        self.stop_button = ttk.Button(
            control_frame,
            text="■ 停止",
            command=self.stop_play
        )

        self.stop_button.pack(
            side=tk.LEFT,
            padx=5
        )

        self.progress = ttk.Progressbar(
            control_frame,
            orient=tk.HORIZONTAL,
            mode="determinate"
        )

        self.progress.pack(
            side=tk.LEFT,
            fill=tk.X,
            expand=True,
            padx=20
        )

        # ====================================================
        # 下方图表
        # ====================================================

        chart_frame = tk.Frame(
            self.root
        )

        chart_frame.pack(
            fill=tk.BOTH,
            expand=True,
            padx=10
        )

        # ----------------------------------------------------
        # 趋势图
        # ----------------------------------------------------

        trend_frame = tk.LabelFrame(
            chart_frame,
            text="实时人流趋势",
            font=("Microsoft YaHei", 12)
        )

        trend_frame.pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True,
            padx=(0, 5)
        )

        self.fig, self.ax = plt.subplots(
            figsize=(7, 3)
        )

        self.canvas = FigureCanvasTkAgg(
            self.fig,
            master=trend_frame
        )

        self.canvas.get_tk_widget().pack(
            fill=tk.BOTH,
            expand=True
        )

        # ----------------------------------------------------
        # 热力图
        # ----------------------------------------------------

        heat_frame = tk.LabelFrame(
            chart_frame,
            text="实时热力图",
            font=("Microsoft YaHei", 12)
        )

        heat_frame.pack(
            side=tk.RIGHT,
            fill=tk.BOTH,
            expand=True,
            padx=(5, 0)
        )

        self.heatmap_label = tk.Label(
            heat_frame,
            bg="black"
        )

        self.heatmap_label.pack(
            fill=tk.BOTH,
            expand=True
        )


    # ========================================================
    # KPI创建
    # ========================================================

    def create_kpi(
        self,
        parent,
        name
    ):

        frame = tk.Frame(
            parent,
            relief=tk.GROOVE,
            borderwidth=1
        )

        frame.pack(
            fill=tk.X,
            padx=10,
            pady=4
        )

        label = tk.Label(
            frame,
            text=name,
            font=("Microsoft YaHei", 10)
        )

        label.pack(
            pady=(5, 0)
        )

        value = tk.Label(
            frame,
            text="0",
            font=("Arial", 18, "bold")
        )

        value.pack(
            pady=(0, 5)
        )

        return value


    # ========================================================
    # 打开视频
    # ========================================================

    def open_video(self):

        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(
            VIDEO_PATH
        )

        if not self.cap.isOpened():
            raise RuntimeError(
                f"无法打开视频：{VIDEO_PATH}"
            )

        self.fps_video = self.cap.get(
            cv2.CAP_PROP_FPS
        )

        if self.fps_video <= 0:
            self.fps_video = 30

        self.total_frames = int(
            self.cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        self.video_width = int(
            self.cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        self.video_height = int(
            self.cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        # 初始化热力图
        self.heatmap = np.zeros(
            (
                self.video_height,
                self.video_width
            ),
            dtype=np.float32
        )

        # 读取第一帧
        ret, frame = self.cap.read()

        if ret:

            self.current_frame_index = 0

            with self.frame_lock:
                self.latest_frame = frame.copy()

            self.display_frame(frame)


    # ========================================================
    # 播放
    # ========================================================

    def start_play(self):

        if self.running:
            self.paused = False
            return

        self.running = True
        self.paused = False
        self.stop_requested = False

        self.worker_thread = threading.Thread(
            target=self.process_video,
            daemon=True
        )

        self.worker_thread.start()


    # ========================================================
    # 暂停
    # ========================================================

    def pause_play(self):

        self.paused = True


    # ========================================================
    # 停止
    # ========================================================

    def stop_play(self):

        self.stop_requested = True
        self.running = False
        self.paused = False


    # ========================================================
    # 重播
    # ========================================================

    def restart_video(self):

        self.stop_requested = True
        self.running = False
        self.paused = False

        time.sleep(0.05)

        self.reset_statistics()

        self.open_video()

        self.start_play()


    # ========================================================
    # 重置统计
    # ========================================================

    def reset_statistics(self):

        with self.data_lock:

            self.current_people = 0
            self.current_ids = []

            self.in_count = 0
            self.out_count = 0

            self.roi_current = 0

            self.roi_visitors.clear()
            self.roi_inside.clear()

            self.roi_visits = 0

            self.track_history.clear()

            self.track_first_side.clear()
            self.track_last_side.clear()

            self.track_frame_count.clear()

            self.history_time.clear()
            self.history_people.clear()
            self.history_in.clear()
            self.history_out.clear()
            self.history_roi.clear()

        with self.heatmap_lock:

            self.heatmap = np.zeros(
                (
                    self.video_height,
                    self.video_width
                ),
                dtype=np.float32
            )


    # ========================================================
    # 视频处理线程
    # ========================================================

    def process_video(self):

        print("视频处理线程启动")

        last_time = time.time()

        while self.running:

            if self.stop_requested:
                break

            if self.paused:

                time.sleep(0.05)
                continue

            ret, frame = self.cap.read()

            if not ret:

                self.running = False

                print("视频播放结束")

                break

            self.current_frame_index += 1

            # ------------------------------------------------
            # YOLO + ByteTrack
            # ------------------------------------------------

            try:

                results = self.model.track(
                    frame,
                    tracker="bytetrack.yaml",
                    persist=True,
                    classes=CLASSES,
                    conf=CONF,
                    device=DEVICE,
                    verbose=False
                )

            except Exception as e:

                print("YOLO推理错误：", e)

                continue

            annotated = frame.copy()

            current_ids = []

            current_roi_ids = set()

            # ------------------------------------------------
            # 处理检测结果
            # ------------------------------------------------

            if results and len(results) > 0:

                result = results[0]

                if result.boxes is not None:

                    boxes = result.boxes

                    if boxes.id is not None:

                        ids = boxes.id.int().cpu().tolist()

                        xyxy = boxes.xyxy.cpu().numpy()

                        for track_id, box in zip(
                            ids,
                            xyxy
                        ):

                            x1, y1, x2, y2 = map(
                                int,
                                box
                            )

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

                            current_ids.append(
                                track_id
                            )

                            # ==================================
                            # Track历史
                            # ==================================

                            if track_id not in self.track_history:

                                self.track_history[
                                    track_id
                                ] = []

                            self.track_history[
                                track_id
                            ].append(
                                point
                            )

                            if len(
                                self.track_history[track_id]
                            ) > 30:

                                self.track_history[
                                    track_id
                                ] = self.track_history[
                                    track_id
                                ][-30:]

                            self.track_frame_count[
                                track_id
                            ] = self.track_frame_count.get(
                                track_id,
                                0
                            ) + 1

                            # ==================================
                            # 计算计数线方向
                            # ==================================

                            side = point_side(
                                point
                            )

                            if track_id not in self.track_first_side:

                                self.track_first_side[
                                    track_id
                                ] = side

                            previous_side = self.track_last_side.get(
                                track_id,
                                side
                            )

                            # ==================================
                            # IN / OUT
                            # ==================================

                            frame_count = self.track_frame_count[
                                track_id
                            ]

                            if (
                                frame_count >= MIN_TRACK_FRAMES
                                and
                                abs(side) > 1
                                and
                                abs(previous_side) > 1
                            ):

                                # 负 -> 正 = IN
                                if (
                                    previous_side < 0
                                    and side > 0
                                ):

                                    movement = abs(
                                        side - previous_side
                                    )

                                    if movement >= MIN_MOVEMENT_DISTANCE:

                                        self.in_count += 1

                                        # 防止同一ID重复计数
                                        self.track_last_side[
                                            track_id
                                        ] = 999999

                                # 正 -> 负 = OUT
                                elif (
                                    previous_side > 0
                                    and side < 0
                                ):

                                    movement = abs(
                                        side - previous_side
                                    )

                                    if movement >= MIN_MOVEMENT_DISTANCE:

                                        self.out_count += 1

                                        self.track_last_side[
                                            track_id
                                        ] = -999999

                            # 正常更新side
                            if abs(
                                self.track_last_side.get(
                                    track_id,
                                    side
                                )
                            ) < 100000:

                                self.track_last_side[
                                    track_id
                                ] = side

                            # ==================================
                            # ROI
                            # ==================================

                            inside = point_in_polygon(
                                point,
                                ROI_POINTS
                            )

                            if inside:

                                current_roi_ids.add(
                                    track_id
                                )

                                self.roi_visitors.add(
                                    track_id
                                )

                                # 首次进入ROI
                                if track_id not in self.roi_inside:

                                    self.roi_inside.add(
                                        track_id
                                    )

                                    self.roi_visits += 1

                            else:

                                self.roi_inside.discard(
                                    track_id
                                )

                            # ==================================
                            # 热力图
                            # ==================================

                            with self.heatmap_lock:

                                if (
                                    0 <= cx < self.video_width
                                    and
                                    0 <= cy < self.video_height
                                ):

                                    self.heatmap[
                                        cy,
                                        cx
                                    ] += 1

                            # ==================================
                            # 绘制检测框
                            # ==================================

                            cv2.rectangle(
                                annotated,
                                (x1, y1),
                                (x2, y2),
                                (0, 255, 0),
                                2
                            )

                            cv2.circle(
                                annotated,
                                point,
                                4,
                                (0, 0, 255),
                                -1
                            )

                            cv2.putText(
                                annotated,
                                f"ID:{track_id}",
                                (x1, max(20, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 0),
                                2
                            )

                            # ==================================
                            # 绘制轨迹
                            # ==================================

                            history = self.track_history[
                                track_id
                            ]

                            for i in range(
                                1,
                                len(history)
                            ):

                                cv2.line(
                                    annotated,
                                    history[i - 1],
                                    history[i],
                                    (255, 255, 0),
                                    2
                                )

            # =================================================
            # 更新ROI人数
            # =================================================

            self.roi_current = len(
                current_roi_ids
            )

            self.current_people = len(
                current_ids
            )

            self.current_ids = current_ids

            # =================================================
            # 绘制计数线
            # =================================================

            cv2.line(
                annotated,
                LINE_P1,
                LINE_P2,
                (0, 0, 255),
                3
            )

            cv2.circle(
                annotated,
                LINE_P1,
                6,
                (255, 0, 0),
                -1
            )

            cv2.circle(
                annotated,
                LINE_P2,
                6,
                (255, 0, 0),
                -1
            )

            cv2.putText(
                annotated,
                "COUNTING LINE",
                (
                    LINE_P1[0],
                    max(25, LINE_P1[1] - 10)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            # =================================================
            # 绘制ROI
            # =================================================

            roi_np = np.array(
                ROI_POINTS,
                dtype=np.int32
            )

            cv2.polylines(
                annotated,
                [roi_np],
                True,
                (255, 0, 255),
                3
            )

            # =================================================
            # 左上角实时信息
            # =================================================

            cv2.rectangle(
                annotated,
                (10, 10),
                (280, 120),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                annotated,
                f"People: {self.current_people}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                annotated,
                f"IN: {self.in_count}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                annotated,
                f"OUT: {self.out_count}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            # =================================================
            # 最新帧
            # =================================================

            with self.frame_lock:

                self.latest_frame = annotated

            # =================================================
            # FPS
            # =================================================

            self.fps_counter += 1

            now = time.time()

            if now - last_time >= 1.0:

                self.process_fps = (
                    self.fps_counter /
                    (now - last_time)
                )

                self.fps_counter = 0
                last_time = now

            # =================================================
            # 趋势数据
            # =================================================

            elapsed = (
                self.current_frame_index /
                self.fps_video
            )

            with self.data_lock:

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
                    self.roi_current
                )

                if len(self.history_time) > MAX_HISTORY:

                    self.history_time = \
                        self.history_time[-MAX_HISTORY:]

                    self.history_people = \
                        self.history_people[-MAX_HISTORY:]

                    self.history_in = \
                        self.history_in[-MAX_HISTORY:]

                    self.history_out = \
                        self.history_out[-MAX_HISTORY:]

                    self.history_roi = \
                        self.history_roi[-MAX_HISTORY:]


        self.running = False


    # ========================================================
    # 视频UI刷新
    # ========================================================

    def update_video_ui(self):

        try:

            with self.frame_lock:

                if self.latest_frame is not None:

                    frame = self.latest_frame.copy()

                else:

                    frame = None

            if frame is not None:

                self.display_frame(
                    frame
                )

                if self.total_frames > 0:

                    value = (
                        self.current_frame_index /
                        self.total_frames
                    ) * 100

                    self.progress["value"] = value

        except Exception as e:

            print("视频UI错误：", e)

        self.root.after(
            VIDEO_UI_INTERVAL,
            self.update_video_ui
        )


    # ========================================================
    # 显示视频
    # ========================================================

    def display_frame(
        self,
        frame
    ):

        # 根据UI尺寸自动缩放
        label_width = self.video_label.winfo_width()
        label_height = self.video_label.winfo_height()

        if (
            label_width <= 10
            or
            label_height <= 10
        ):

            label_width = 900
            label_height = 500

        h, w = frame.shape[:2]

        scale = min(
            label_width / w,
            label_height / h
        )

        new_w = max(
            1,
            int(w * scale)
        )

        new_h = max(
            1,
            int(h * scale)
        )

        frame = cv2.resize(
            frame,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA
        )

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(
            frame
        )

        photo = ImageTk.PhotoImage(
            image=image
        )

        self.video_label.configure(
            image=photo
        )

        self.video_label.image = photo


    # ========================================================
    # KPI刷新
    # ========================================================

    def update_kpi(self):

        try:

            with self.data_lock:

                current_people = self.current_people
                in_count = self.in_count
                out_count = self.out_count

                roi_current = self.roi_current
                roi_visitors = len(
                    self.roi_visitors
                )

                roi_visits = self.roi_visits

                ids = self.current_ids.copy()

            self.kpi_current.config(
                text=str(current_people)
            )

            self.kpi_in.config(
                text=str(in_count)
            )

            self.kpi_out.config(
                text=str(out_count)
            )

            self.kpi_total.config(
                text=str(
                    in_count + out_count
                )
            )

            self.kpi_roi_current.config(
                text=str(roi_current)
            )

            self.kpi_roi_visitors.config(
                text=str(roi_visitors)
            )

            self.kpi_roi_visits.config(
                text=str(roi_visits)
            )

            self.kpi_fps.config(
                text=f"{self.process_fps:.1f}"
            )

            if ids:

                id_text = (
                    "Track IDs："
                    + ", ".join(
                        map(str, ids[:20])
                    )
                )

                if len(ids) > 20:

                    id_text += " ..."

            else:

                id_text = "Track IDs：-"

            self.kpi_ids.config(
                text=id_text
            )

        except Exception as e:

            print("KPI更新错误：", e)

        self.root.after(
            KPI_UI_INTERVAL,
            self.update_kpi
        )


    # ========================================================
    # 趋势图
    # ========================================================

    def update_chart(self):

        try:

            with self.data_lock:

                times = self.history_time.copy()
                people = self.history_people.copy()
                in_data = self.history_in.copy()
                out_data = self.history_out.copy()
                roi_data = self.history_roi.copy()

            self.ax.clear()

            if times:

                self.ax.plot(
                    times,
                    people,
                    label="当前人数",
                    linewidth=2
                )

                self.ax.plot(
                    times,
                    in_data,
                    label="累计进入",
                    linewidth=2
                )

                self.ax.plot(
                    times,
                    out_data,
                    label="累计离开",
                    linewidth=2
                )

                self.ax.plot(
                    times,
                    roi_data,
                    label="ROI人数",
                    linewidth=2
                )

            self.ax.set_title(
                "实时人流变化趋势"
            )

            self.ax.set_xlabel(
                "时间 / 秒"
            )

            self.ax.set_ylabel(
                "人数"
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

        except Exception as e:

            print("趋势图更新错误：", e)

        self.root.after(
            CHART_UI_INTERVAL,
            self.update_chart
        )


    # ========================================================
    # 热力图
    # ========================================================

    def update_heatmap(self):

        try:

            with self.heatmap_lock:

                heat = self.heatmap.copy()

            if heat.max() > 0:

                kernel = HEATMAP_BLUR_KERNEL

                if kernel % 2 == 0:
                    kernel += 1

                heat = cv2.GaussianBlur(
                    heat,
                    (kernel, kernel),
                    0
                )

                heat_norm = cv2.normalize(
                    heat,
                    None,
                    0,
                    255,
                    cv2.NORM_MINMAX
                )

                heat_uint8 = heat_norm.astype(
                    np.uint8
                )

                color_heatmap = cv2.applyColorMap(
                    heat_uint8,
                    cv2.COLORMAP_JET
                )

                # ------------------------------------------------
                # 叠加ROI
                # ------------------------------------------------

                roi_np = np.array(
                    ROI_POINTS,
                    dtype=np.int32
                )

                cv2.polylines(
                    color_heatmap,
                    [roi_np],
                    True,
                    (255, 255, 255),
                    2
                )

                # ------------------------------------------------
                # 叠加计数线
                # ------------------------------------------------

                cv2.line(
                    color_heatmap,
                    LINE_P1,
                    LINE_P2,
                    (255, 255, 255),
                    2
                )

                color_heatmap = cv2.cvtColor(
                    color_heatmap,
                    cv2.COLOR_BGR2RGB
                )

                image = Image.fromarray(
                    color_heatmap
                )

                # 热力图尺寸
                width = 500
                height = int(
                    image.height *
                    width /
                    image.width
                )

                image = image.resize(
                    (width, height),
                    Image.Resampling.LANCZOS
                )

                photo = ImageTk.PhotoImage(
                    image=image
                )

                self.heatmap_label.config(
                    image=photo
                )

                self.heatmap_label.image = photo

        except Exception as e:

            print("热力图更新错误：", e)

        self.root.after(
            HEATMAP_UI_INTERVAL,
            self.update_heatmap
        )


    # ========================================================
    # 关闭
    # ========================================================

    def on_close(self):

        print("正在关闭...")

        self.running = False
        self.stop_requested = True

        if self.cap is not None:

            self.cap.release()

        plt.close(
            self.fig
        )

        self.root.destroy()


# ============================================================
# 5. 程序入口
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("人流进出计数与区域热度分析")
    print("Realtime Dashboard V12.3")
    print("=" * 60)

    print("视频：", VIDEO_PATH)
    print("模型：", MODEL_PATH)
    print("设备：", DEVICE)

    root = tk.Tk()

    app = RealtimeDashboard(
        root
    )

    root.mainloop()