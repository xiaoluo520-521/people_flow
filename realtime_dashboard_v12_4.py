import os
import json
import time
import threading
import ctypes
from collections import deque

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from ultralytics import YOLO


# ============================================================
# 1. Windows DPI 适配
# ============================================================

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


# ============================================================
# 2. 路径
# ============================================================

BASE_DIR = r"E:\people_flow"

VIDEO_PATH = os.path.join(
    BASE_DIR,
    "videos",
    "test.mp4"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "yolo11n.pt"
)

CONFIG_PATH = os.path.join(
    BASE_DIR,
    "config.json"
)


# ============================================================
# 3. 参数
# ============================================================

WINDOW_WIDTH = 1500
WINDOW_HEIGHT = 900

FPS_UPDATE = 30

HISTORY_LENGTH = 180

HEATMAP_BLUR = 51

DEVICE = "cuda:0"

CONF_THRESHOLD = 0.30

PERSON_CLASS = 0


# ============================================================
# 4. 读取配置
# ============================================================

def load_config():

    if not os.path.exists(CONFIG_PATH):

        raise FileNotFoundError(
            f"找不到配置文件：{CONFIG_PATH}"
        )

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        config = json.load(f)

    return config


# ============================================================
# 5. 判断点在线的哪一侧
# ============================================================

def side_of_line(
    point,
    p1,
    p2
):

    x, y = point

    x1, y1 = p1
    x2, y2 = p2

    value = (
        (x2 - x1) * (y - y1)
        -
        (y2 - y1) * (x - x1)
    )

    if value > 0:
        return 1

    elif value < 0:
        return -1

    return 0


# ============================================================
# 6. 判断点是否在 ROI 内
# ============================================================

def point_in_polygon(
    point,
    polygon
):

    if (
        polygon is None
        or len(polygon) < 3
    ):

        return False

    contour = np.array(
        polygon,
        dtype=np.int32
    )

    result = cv2.pointPolygonTest(
        contour,
        (
            float(point[0]),
            float(point[1])
        ),
        False
    )

    return result >= 0


# ============================================================
# 7. 主类
# ============================================================

class RealtimeDashboardV13_3:

    def __init__(
        self,
        root
    ):

        self.root = root

        self.root.title(
            "人流进出计数与区域热度分析 - V13.3"
        )

        self.root.geometry(
            f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}"
        )

        self.root.minsize(
            1100,
            700
        )

        self.root.configure(
            bg="#202020"
        )

        # ====================================================
        # 读取配置
        # ====================================================

        self.config = load_config()

        self.line_p1 = tuple(
            self.config[
                "counting_line"
            ][
                "p1"
            ]
        )

        self.line_p2 = tuple(
            self.config[
                "counting_line"
            ][
                "p2"
            ]
        )

        self.roi_polygon = [
            tuple(point)
            for point in self.config[
                "roi"
            ]
        ]

        # ====================================================
        # 视频参数
        # ====================================================

        self.cap = None

        self.video_width = 1280

        self.video_height = 720

        self.video_fps = 30

        self.total_frames = 0

        self.current_frame = 0

        # ====================================================
        # YOLO
        # ====================================================

        self.model = None

        # ====================================================
        # 线程控制
        # ====================================================

        self.running = True

        self.paused = False

        self.restart_requested = False

        self.lock = threading.Lock()

        # ====================================================
        # 最新数据
        # ====================================================

        self.latest_frame = None

        self.latest_tracks = []

        self.latest_frame_id = -1

        self.processed_frame_id = -1

        # ====================================================
        # 计数
        # ====================================================

        self.track_last_side = {}

        self.counted_ids = set()

        self.in_count = 0

        self.out_count = 0

        # ====================================================
        # ROI
        # ====================================================

        self.roi_inside = set()

        self.roi_visitors = set()

        self.roi_visits = 0

        # ====================================================
        # 热力图
        # ====================================================

        self.heat_points = []

        # ====================================================
        # 趋势数据
        # ====================================================

        self.trend_people = deque(
            maxlen=HISTORY_LENGTH
        )

        self.trend_in = deque(
            maxlen=HISTORY_LENGTH
        )

        self.trend_out = deque(
            maxlen=HISTORY_LENGTH
        )

        # ====================================================
        # 创建 UI
        # ====================================================

        self.create_ui()

        # ====================================================
        # 打开视频
        # ====================================================

        self.open_video()

        # ====================================================
        # 加载 YOLO
        # ====================================================

        self.create_model()

        # ====================================================
        # 启动推理线程
        # ====================================================

        self.worker_thread = threading.Thread(
            target=self.inference_worker,
            daemon=True
        )

        self.worker_thread.start()

        # ====================================================
        # UI 定时刷新
        # ====================================================

        self.root.after(
            FPS_UPDATE,
            self.update_ui
        )

        # ====================================================
        # 关闭窗口
        # ====================================================

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.close
        )


    # ========================================================
    # 创建模型
    # ========================================================

    def create_model(self):

        print("=" * 60)

        print(
            "正在加载 YOLO 模型..."
        )

        print(
            f"模型：{MODEL_PATH}"
        )

        print(
            f"设备：{DEVICE}"
        )

        self.model = YOLO(
            MODEL_PATH
        )

        print(
            "YOLO 模型加载完成"
        )

        print("=" * 60)


    # ========================================================
    # 打开视频
    # ========================================================

    def open_video(self):

        if self.cap is not None:

            try:

                self.cap.release()

            except Exception:

                pass

        self.cap = cv2.VideoCapture(
            VIDEO_PATH
        )

        if not self.cap.isOpened():

            raise RuntimeError(
                f"无法打开视频：{VIDEO_PATH}"
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

        print(
            f"视频："
            f"{self.video_width}x"
            f"{self.video_height}"
        )

        print(
            f"FPS：{self.video_fps:.2f}"
        )

        print(
            f"总帧数：{self.total_frames}"
        )


    # ========================================================
    # 创建 UI
    # ========================================================

    def create_ui(self):

        # ====================================================
        # 根窗口 Grid
        # ====================================================

        self.root.grid_rowconfigure(
            0,
            weight=0
        )

        self.root.grid_rowconfigure(
            1,
            weight=5
        )

        self.root.grid_rowconfigure(
            2,
            weight=0
        )

        self.root.grid_rowconfigure(
            3,
            weight=3
        )

        self.root.grid_columnconfigure(
            0,
            weight=7
        )

        self.root.grid_columnconfigure(
            1,
            weight=3
        )

        # ====================================================
        # 标题
        # ====================================================

        title_frame = tk.Frame(
            self.root,
            bg="#202020",
            height=45
        )

        title_frame.grid(
            row=0,
            column=0,
            columnspan=2,
            sticky="nsew"
        )

        title_frame.grid_propagate(
            False
        )

        title_label = tk.Label(
            title_frame,
            text="人流进出计数与区域热度分析",
            font=(
                "Microsoft YaHei",
                20,
                "bold"
            ),
            fg="white",
            bg="#202020"
        )

        title_label.pack(
            expand=True
        )

        # ====================================================
        # 视频区域
        # ====================================================

        self.video_frame = tk.Frame(
            self.root,
            bg="black",
            bd=1,
            relief="solid"
        )

        self.video_frame.grid(
            row=1,
            column=0,
            sticky="nsew",
            padx=(8, 4),
            pady=(2, 4)
        )

        self.video_frame.grid_rowconfigure(
            0,
            weight=1
        )

        self.video_frame.grid_columnconfigure(
            0,
            weight=1
        )

        self.video_canvas = tk.Canvas(
            self.video_frame,
            bg="black",
            highlightthickness=0
        )

        self.video_canvas.grid(
            row=0,
            column=0,
            sticky="nsew"
        )

        # ====================================================
        # KPI
        # ====================================================

        self.kpi_frame = tk.Frame(
            self.root,
            bg="#292929",
            bd=1,
            relief="solid"
        )

        self.kpi_frame.grid(
            row=1,
            column=1,
            sticky="nsew",
            padx=(4, 8),
            pady=(2, 4)
        )

        self.kpi_frame.grid_columnconfigure(
            0,
            weight=1
        )

        self.create_kpi_ui()

        # ====================================================
        # 控制栏
        # ====================================================

        self.control_frame = tk.Frame(
            self.root,
            bg="#303030",
            height=48
        )

        self.control_frame.grid(
            row=2,
            column=0,
            columnspan=2,
            sticky="nsew",
            padx=8,
            pady=2
        )

        self.control_frame.grid_propagate(
            False
        )

        self.create_control_ui()

        # ====================================================
        # 底部左侧
        # ====================================================

        self.bottom_left = tk.Frame(
            self.root,
            bg="#202020",
            bd=1,
            relief="solid"
        )

        self.bottom_left.grid(
            row=3,
            column=0,
            sticky="nsew",
            padx=(8, 4),
            pady=(4, 8)
        )

        # ====================================================
        # 底部右侧
        # ====================================================

        self.bottom_right = tk.Frame(
            self.root,
            bg="#202020",
            bd=1,
            relief="solid"
        )

        self.bottom_right.grid(
            row=3,
            column=1,
            sticky="nsew",
            padx=(4, 8),
            pady=(4, 8)
        )

        # ====================================================
        # 底部自适应
        # ====================================================

        self.bottom_left.grid_rowconfigure(
            0,
            weight=1
        )

        self.bottom_left.grid_columnconfigure(
            0,
            weight=1
        )

        self.bottom_right.grid_rowconfigure(
            0,
            weight=1
        )

        self.bottom_right.grid_columnconfigure(
            0,
            weight=1
        )

        # ====================================================
        # 趋势图 Canvas
        # ====================================================

        self.trend_canvas = tk.Canvas(
            self.bottom_left,
            bg="#151515",
            highlightthickness=0
        )

        self.trend_canvas.grid(
            row=0,
            column=0,
            sticky="nsew"
        )

        # ====================================================
        # 热力图 Canvas
        # ====================================================

        self.heatmap_canvas = tk.Canvas(
            self.bottom_right,
            bg="#151515",
            highlightthickness=0
        )

        self.heatmap_canvas.grid(
            row=0,
            column=0,
            sticky="nsew"
        )


    # ========================================================
    # KPI UI
    # ========================================================

    def create_kpi_ui(self):

        # ====================================================
        # 标题
        # ====================================================

        title = tk.Label(
            self.kpi_frame,
            text="实时统计",
            font=(
                "Microsoft YaHei",
                16,
                "bold"
            ),
            fg="white",
            bg="#292929"
        )

        title.grid(
            row=0,
            column=0,
            pady=(8, 4)
        )

        # ====================================================
        # KPI 标签
        # ====================================================

        self.kpi_labels = {}

        items = [
            (
                "当前人数",
                "current_people"
            ),
            (
                "进入人数",
                "in_count"
            ),
            (
                "离开人数",
                "out_count"
            ),
            (
                "ROI当前人数",
                "roi_current"
            ),
            (
                "ROI访问人数",
                "roi_visitors"
            ),
            (
                "ROI访问次数",
                "roi_visits"
            ),
            (
                "当前跟踪ID",
                "track_ids"
            ),
            (
                "视频进度",
                "progress"
            )
        ]

        # ====================================================
        # 让每一行平均分配高度
        # ====================================================

        self.kpi_frame.grid_rowconfigure(
            0,
            weight=0
        )

        for i in range(
            len(items)
        ):

            self.kpi_frame.grid_rowconfigure(
                i + 1,
                weight=1
            )

        self.kpi_frame.grid_columnconfigure(
            0,
            weight=1
        )

        # ====================================================
        # 创建 KPI 项
        # ====================================================

        for i, (name, key) in enumerate(
            items
        ):

            frame = tk.Frame(
                self.kpi_frame,
                bg="#333333"
            )

            frame.grid(
                row=i + 1,
                column=0,
                sticky="nsew",
                padx=12,
                pady=2
            )

            frame.grid_columnconfigure(
                0,
                weight=1
            )

            frame.grid_columnconfigure(
                1,
                weight=1
            )

            # ------------------------------------------------
            # 名称
            # ------------------------------------------------

            label = tk.Label(
                frame,
                text=name,
                font=(
                    "Microsoft YaHei",
                    10
                ),
                fg="#cccccc",
                bg="#333333",
                anchor="w"
            )

            label.grid(
                row=0,
                column=0,
                padx=10,
                sticky="w"
            )

            # ------------------------------------------------
            # 数值
            # ------------------------------------------------

            value = tk.Label(
                frame,
                text="0",
                font=(
                    "Microsoft YaHei",
                    12,
                    "bold"
                ),
                fg="white",
                bg="#333333",
                anchor="e"
            )

            value.grid(
                row=0,
                column=1,
                padx=10,
                sticky="e"
            )

            self.kpi_labels[
                key
            ] = value


    # ========================================================
    # 控制栏
    # ========================================================

    def create_control_ui(self):

        self.play_button = ttk.Button(
            self.control_frame,
            text="暂停",
            command=self.toggle_pause
        )

        self.play_button.pack(
            side="left",
            padx=(10, 5),
            pady=8
        )

        self.restart_button = ttk.Button(
            self.control_frame,
            text="重新播放",
            command=self.restart_video
        )

        self.restart_button.pack(
            side="left",
            padx=5,
            pady=8
        )

        self.progress_label = tk.Label(
            self.control_frame,
            text="0 / 0",
            font=(
                "Microsoft YaHei",
                10
            ),
            fg="white",
            bg="#303030"
        )

        self.progress_label.pack(
            side="left",
            padx=15
        )

        self.status_label = tk.Label(
            self.control_frame,
            text="运行中",
            font=(
                "Microsoft YaHei",
                10
            ),
            fg="#80ff80",
            bg="#303030"
        )

        self.status_label.pack(
            side="right",
            padx=15
        )


    # ========================================================
    # 暂停 / 播放
    # ========================================================

    def toggle_pause(self):

        self.paused = not self.paused

        if self.paused:

            self.play_button.config(
                text="播放"
            )

            self.status_label.config(
                text="已暂停",
                fg="#ffff80"
            )

        else:

            self.play_button.config(
                text="暂停"
            )

            self.status_label.config(
                text="运行中",
                fg="#80ff80"
            )


    # ========================================================
    # 重新播放
    # ========================================================

    def restart_video(self):

        print(
            "请求重新播放..."
        )

        with self.lock:

            self.restart_requested = True

        self.paused = False

        self.play_button.config(
            text="暂停"
        )

        self.status_label.config(
            text="重新播放中...",
            fg="#80ffff"
        )


    # ========================================================
    # 重置统计
    # ========================================================

    def reset_statistics(self):

        self.track_last_side.clear()

        self.counted_ids.clear()

        self.in_count = 0

        self.out_count = 0

        self.roi_inside.clear()

        self.roi_visitors.clear()

        self.roi_visits = 0

        self.heat_points.clear()

        self.trend_people.clear()

        self.trend_in.clear()

        self.trend_out.clear()

        self.latest_tracks = []

        self.latest_frame_id = -1

        self.processed_frame_id = -1


    # ========================================================
    # 后台推理线程
    # ========================================================

    def inference_worker(self):

        while self.running:

            # =================================================
            # 重新播放
            # =================================================

            if self.restart_requested:

                print(
                    "正在重新初始化视频和模型..."
                )

                self.cap.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    0
                )

                self.current_frame = 0

                self.reset_statistics()

                try:

                    self.model = YOLO(
                        MODEL_PATH
                    )

                except Exception as e:

                    print(
                        "模型重新加载失败：",
                        e
                    )

                with self.lock:

                    self.restart_requested = False

                self.status_label_safe(
                    "运行中",
                    "#80ff80"
                )

                print(
                    "重新播放完成"
                )

            # =================================================
            # 暂停
            # =================================================

            if self.paused:

                time.sleep(
                    0.05
                )

                continue

            # =================================================
            # 视频
            # =================================================

            if self.cap is None:

                time.sleep(
                    0.05
                )

                continue

            ret, frame = self.cap.read()

            if not ret:

                self.paused = True

                self.status_label_safe(
                    "播放结束",
                    "#ffff80"
                )

                self.play_button_safe(
                    "播放"
                )

                time.sleep(
                    0.1
                )

                continue

            self.current_frame += 1

            frame_id = (
                self.current_frame
            )

            # =================================================
            # YOLO + ByteTrack
            # =================================================

            try:

                results = self.model.track(
                    frame,
                    persist=True,
                    classes=[
                        PERSON_CLASS
                    ],
                    conf=CONF_THRESHOLD,
                    device=DEVICE,
                    tracker="bytetrack.yaml",
                    verbose=False
                )

            except Exception as e:

                print(
                    "YOLO 推理错误：",
                    e
                )

                time.sleep(
                    0.05
                )

                continue

            tracks = []

            # =================================================
            # 解析结果
            # =================================================

            if (
                results
                and len(results) > 0
            ):

                result = results[0]

                if (
                    result.boxes is not None
                    and len(result.boxes) > 0
                ):

                    boxes = result.boxes

                    xyxy = (
                        boxes.xyxy
                        .cpu()
                        .numpy()
                    )

                    ids = None

                    if boxes.id is not None:

                        ids = (
                            boxes.id
                            .cpu()
                            .numpy()
                            .astype(int)
                        )

                    if ids is not None:

                        for box, track_id in zip(
                            xyxy,
                            ids
                        ):

                            x1, y1, x2, y2 = box

                            cx = int(
                                (x1 + x2)
                                / 2
                            )

                            cy = int(
                                (y1 + y2)
                                / 2
                            )

                            tracks.append(
                                {
                                    "id": int(
                                        track_id
                                    ),
                                    "bbox": (
                                        int(x1),
                                        int(y1),
                                        int(x2),
                                        int(y2)
                                    ),
                                    "center": (
                                        cx,
                                        cy
                                    )
                                }
                            )

            # =================================================
            # 保存最新数据
            # =================================================

            with self.lock:

                self.latest_frame = (
                    frame.copy()
                )

                self.latest_tracks = (
                    tracks
                )

                self.latest_frame_id = (
                    frame_id
                )

            time.sleep(
                0.001
            )


    # ========================================================
    # 安全修改状态
    # ========================================================

    def status_label_safe(
        self,
        text_value,
        color
    ):

        try:

            self.root.after(
                0,
                lambda: self.status_label.config(
                    text=text_value,
                    fg=color
                )
            )

        except Exception:

            pass


    def play_button_safe(
        self,
        text_value
    ):

        try:

            self.root.after(
                0,
                lambda: self.play_button.config(
                    text=text_value
                )
            )

        except Exception:

            pass


    # ========================================================
    # 处理跟踪
    # ========================================================

    def process_tracks(
        self,
        tracks
    ):

        current_ids = set()

        for track in tracks:

            track_id = track[
                "id"
            ]

            point = track[
                "center"
            ]

            current_ids.add(
                track_id
            )

            # =================================================
            # 计数线方向
            # =================================================

            side = side_of_line(
                point,
                self.line_p1,
                self.line_p2
            )

            if (
                track_id
                not in self.track_last_side
            ):

                self.track_last_side[
                    track_id
                ] = side

            else:

                previous_side = (
                    self.track_last_side[
                        track_id
                    ]
                )

                if (
                    previous_side != 0
                    and
                    side != 0
                    and
                    previous_side != side
                    and
                    track_id not in self.counted_ids
                ):

                    if (
                        previous_side == -1
                        and
                        side == 1
                    ):

                        self.in_count += 1

                    elif (
                        previous_side == 1
                        and
                        side == -1
                    ):

                        self.out_count += 1

                    self.counted_ids.add(
                        track_id
                    )

                self.track_last_side[
                    track_id
                ] = side

            # =================================================
            # ROI
            # =================================================

            inside = point_in_polygon(
                point,
                self.roi_polygon
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

            # =================================================
            # 热力点
            # =================================================

            self.heat_points.append(
                point
            )

        # =====================================================
        # 清理 ROI
        # =====================================================

        self.roi_inside.intersection_update(
            current_ids
        )

        # =====================================================
        # 限制热力点数量
        # =====================================================

        max_heat_points = 12000

        if (
            len(self.heat_points)
            > max_heat_points
        ):

            self.heat_points = (
                self.heat_points[
                    -max_heat_points:
                ]
            )

        # =====================================================
        # 趋势
        # =====================================================

        self.trend_people.append(
            len(current_ids)
        )

        self.trend_in.append(
            self.in_count
        )

        self.trend_out.append(
            self.out_count
        )


    # ========================================================
    # 更新界面
    # ========================================================

    def update_ui(self):

        if not self.running:

            return

        with self.lock:

            frame = (
                self.latest_frame.copy()
                if self.latest_frame is not None
                else None
            )

            tracks = list(
                self.latest_tracks
            )

            frame_id = (
                self.latest_frame_id
            )

        # ====================================================
        # 只处理新帧
        # ====================================================

        if (
            frame_id != -1
            and
            frame_id
            != self.processed_frame_id
        ):

            self.process_tracks(
                tracks
            )

            self.processed_frame_id = (
                frame_id
            )

        # ====================================================
        # 视频
        # ====================================================

        if frame is not None:

            display_frame = (
                self.draw_frame(
                    frame,
                    tracks
                )
            )

            self.display_video(
                display_frame
            )

        # ====================================================
        # KPI
        # ====================================================

        self.update_kpi(
            tracks
        )

        # ====================================================
        # 趋势
        # ====================================================

        self.draw_trend()

        # ====================================================
        # 热力图
        # ====================================================

        self.draw_heatmap()

        # ====================================================
        # 下一次刷新
        # ====================================================

        self.root.after(
            FPS_UPDATE,
            self.update_ui
        )


    # ========================================================
    # 绘制视频
    # ========================================================

    def draw_frame(
        self,
        frame,
        tracks
    ):

        output = frame.copy()

        # ====================================================
        # ROI
        # ====================================================

        if (
            self.roi_polygon is not None
            and
            len(self.roi_polygon) >= 3
        ):

            polygon = np.array(
                self.roi_polygon,
                dtype=np.int32
            )

            cv2.polylines(
                output,
                [polygon],
                True,
                (0, 255, 255),
                2
            )

        # ====================================================
        # 计数线
        # ====================================================

        cv2.line(
            output,
            self.line_p1,
            self.line_p2,
            (0, 255, 0),
            3
        )

        cv2.circle(
            output,
            self.line_p1,
            6,
            (0, 255, 0),
            -1
        )

        cv2.circle(
            output,
            self.line_p2,
            6,
            (0, 255, 0),
            -1
        )

        # ====================================================
        # 跟踪框
        # ====================================================

        for track in tracks:

            x1, y1, x2, y2 = (
                track["bbox"]
            )

            cx, cy = (
                track["center"]
            )

            track_id = track["id"]

            inside = point_in_polygon(
                (cx, cy),
                self.roi_polygon
            )

            if inside:

                box_color = (
                    0,
                    255,
                    255
                )

            else:

                box_color = (
                    255,
                    180,
                    0
                )

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                box_color,
                2
            )

            cv2.circle(
                output,
                (cx, cy),
                4,
                box_color,
                -1
            )

            cv2.putText(
                output,
                f"ID {track_id}",
                (
                    x1,
                    max(
                        20,
                        y1 - 7
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                box_color,
                2
            )

        # ====================================================
        # 左上统计
        # ====================================================

        overlay = output.copy()

        cv2.rectangle(
            overlay,
            (10, 10),
            (270, 125),
            (0, 0, 0),
            -1
        )

        output = cv2.addWeighted(
            overlay,
            0.55,
            output,
            0.45,
            0
        )

        current_people = len(
            tracks
        )

        text_lines = [
            f"People: {current_people}",
            f"IN: {self.in_count}",
            f"OUT: {self.out_count}",
            f"ROI: {len(self.roi_inside)}"
        ]

        for i, text_value in enumerate(
            text_lines
        ):

            cv2.putText(
                output,
                text_value,
                (
                    20,
                    35 + i * 25
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

        return output


    # ========================================================
    # 显示视频
    # ========================================================

    def display_video(
        self,
        frame
    ):

        canvas = (
            self.video_canvas
        )

        canvas.update_idletasks()

        canvas_width = (
            canvas.winfo_width()
        )

        canvas_height = (
            canvas.winfo_height()
        )

        if (
            canvas_width <= 1
            or
            canvas_height <= 1
        ):

            return

        # ====================================================
        # 直接填满 Canvas
        # 不保持比例
        # 避免左右黑边
        # ====================================================

        resized = cv2.resize(
            frame,
            (
                canvas_width,
                canvas_height
            ),
            interpolation=cv2.INTER_LINEAR
        )

        rgb = cv2.cvtColor(
            resized,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(
            rgb
        )

        photo = ImageTk.PhotoImage(
            image=image
        )

        canvas.delete(
            "all"
        )

        canvas.create_image(
            0,
            0,
            image=photo,
            anchor="nw"
        )

        canvas.image = photo


    # ========================================================
    # 更新 KPI
    # ========================================================

    def update_kpi(
        self,
        tracks
    ):

        current_people = len(
            tracks
        )

        self.kpi_labels[
            "current_people"
        ].config(
            text=str(
                current_people
            )
        )

        self.kpi_labels[
            "in_count"
        ].config(
            text=str(
                self.in_count
            )
        )

        self.kpi_labels[
            "out_count"
        ].config(
            text=str(
                self.out_count
            )
        )

        self.kpi_labels[
            "roi_current"
        ].config(
            text=str(
                len(
                    self.roi_inside
                )
            )
        )

        self.kpi_labels[
            "roi_visitors"
        ].config(
            text=str(
                len(
                    self.roi_visitors
                )
            )
        )

        self.kpi_labels[
            "roi_visits"
        ].config(
            text=str(
                self.roi_visits
            )
        )

        self.kpi_labels[
            "track_ids"
        ].config(
            text=str(
                len(
                    tracks
                )
            )
        )

        self.kpi_labels[
            "progress"
        ].config(
            text=(
                f"{self.current_frame} / "
                f"{self.total_frames}"
            )
        )

        self.progress_label.config(
            text=(
                f"视频进度："
                f"{self.current_frame} / "
                f"{self.total_frames}"
            )
        )


    # ========================================================
    # 趋势图
    # ========================================================

    def draw_trend(self):

        canvas = (
            self.trend_canvas
        )

        canvas.update_idletasks()

        width = (
            canvas.winfo_width()
        )

        height = (
            canvas.winfo_height()
        )

        if (
            width <= 10
            or
            height <= 10
        ):

            return

        canvas.delete(
            "all"
        )

        # ====================================================
        # 标题
        # ====================================================

        canvas.create_text(
            15,
            15,
            anchor="w",
            text="实时人数趋势",
            fill="white",
            font=(
                "Microsoft YaHei",
                13,
                "bold"
            )
        )

        # ====================================================
        # 坐标轴
        # ====================================================

        left = 55

        right = width - 20

        top = 45

        bottom = height - 35

        canvas.create_line(
            left,
            top,
            left,
            bottom,
            fill="#777777"
        )

        canvas.create_line(
            left,
            bottom,
            right,
            bottom,
            fill="#777777"
        )

        # ====================================================
        # 数据
        # ====================================================

        people = list(
            self.trend_people
        )

        in_data = list(
            self.trend_in
        )

        out_data = list(
            self.trend_out
        )

        if len(people) < 2:

            canvas.create_text(
                width / 2,
                height / 2,
                text="等待数据...",
                fill="#888888",
                font=(
                    "Microsoft YaHei",
                    12
                )
            )

            return

        max_value = max(
            max(people),
            max(in_data),
            max(out_data),
            5
        )

        # ====================================================
        # 网格
        # ====================================================

        for i in range(5):

            y = (
                bottom
                -
                (
                    bottom - top
                )
                * i
                / 4
            )

            value = (
                max_value
                * i
                / 4
            )

            canvas.create_line(
                left,
                y,
                right,
                y,
                fill="#333333"
            )

            canvas.create_text(
                left - 8,
                y,
                text=str(
                    int(value)
                ),
                fill="#999999",
                anchor="e",
                font=(
                    "Arial",
                    8
                )
            )

        # ====================================================
        # 曲线
        # ====================================================

        def make_points(
            data
        ):

            if len(data) < 2:

                return []

            points = []

            n = len(data)

            for i, value in enumerate(
                data
            ):

                x = (
                    left
                    +
                    (
                        right - left
                    )
                    * i
                    /
                    max(
                        1,
                        n - 1
                    )
                )

                y = (
                    bottom
                    -
                    (
                        bottom - top
                    )
                    * value
                    /
                    max_value
                )

                points.append(
                    (
                        x,
                        y
                    )
                )

            return points

        people_points = make_points(
            people
        )

        in_points = make_points(
            in_data
        )

        out_points = make_points(
            out_data
        )

        if len(
            people_points
        ) >= 2:

            canvas.create_line(
                people_points,
                fill="#00ffcc",
                width=2,
                smooth=True
            )

        if len(
            in_points
        ) >= 2:

            canvas.create_line(
                in_points,
                fill="#00ff00",
                width=2,
                smooth=True
            )

        if len(
            out_points
        ) >= 2:

            canvas.create_line(
                out_points,
                fill="#ff6666",
                width=2,
                smooth=True
            )

        # ====================================================
        # 图例
        # ====================================================

        legend_y = (
            height - 15
        )

        canvas.create_text(
            80,
            legend_y,
            text="● 当前人数",
            fill="#00ffcc",
            font=(
                "Microsoft YaHei",
                9
            )
        )

        canvas.create_text(
            190,
            legend_y,
            text="● 累计进入",
            fill="#00ff00",
            font=(
                "Microsoft YaHei",
                9
            )
        )

        canvas.create_text(
            300,
            legend_y,
            text="● 累计离开",
            fill="#ff6666",
            font=(
                "Microsoft YaHei",
                9
            )
        )


    # ========================================================
    # 热力图
    # ========================================================

    def draw_heatmap(self):

        canvas = (
            self.heatmap_canvas
        )

        canvas.update_idletasks()

        width = (
            canvas.winfo_width()
        )

        height = (
            canvas.winfo_height()
        )

        if (
            width <= 10
            or
            height <= 10
        ):

            return

        canvas.delete(
            "all"
        )

        # ====================================================
        # 标题
        # ====================================================

        canvas.create_text(
            15,
            15,
            anchor="w",
            text="实时区域热度",
            fill="white",
            font=(
                "Microsoft YaHei",
                13,
                "bold"
            )
        )

        # ====================================================
        # 热力图
        # ====================================================

        heat = np.zeros(
            (
                self.video_height,
                self.video_width
            ),
            dtype=np.float32
        )

        for x, y in self.heat_points:

            if (
                0 <= x
                < self.video_width
                and
                0 <= y
                < self.video_height
            ):

                heat[
                    int(y),
                    int(x)
                ] += 1

        # ====================================================
        # 高斯模糊
        # ====================================================

        if np.max(
            heat
        ) > 0:

            kernel = (
                HEATMAP_BLUR
            )

            if kernel % 2 == 0:

                kernel += 1

            heat = cv2.GaussianBlur(
                heat,
                (
                    kernel,
                    kernel
                ),
                0
            )

            max_heat = np.max(
                heat
            )

            if max_heat > 0:

                heat = (
                    heat
                    /
                    max_heat
                    *
                    255
                ).astype(
                    np.uint8
                )

            heat_color = cv2.applyColorMap(
                heat,
                cv2.COLORMAP_JET
            )

            # =================================================
            # 调整尺寸
            # =================================================

            display_height = (
                height - 35
            )

            heat_color = cv2.resize(
                heat_color,
                (
                    width,
                    display_height
                ),
                interpolation=cv2.INTER_LINEAR
            )

            heat_img = Image.fromarray(
                cv2.cvtColor(
                    heat_color,
                    cv2.COLOR_BGR2RGB
                )
            )

            photo = ImageTk.PhotoImage(
                heat_img
            )

            canvas.create_image(
                0,
                35,
                image=photo,
                anchor="nw"
            )

            canvas.image = photo

        else:

            canvas.create_text(
                width / 2,
                height / 2,
                text="等待热力数据...",
                fill="#777777",
                font=(
                    "Microsoft YaHei",
                    12
                )
            )


    # ========================================================
    # 关闭
    # ========================================================

    def close(self):

        print(
            "正在关闭程序..."
        )

        self.running = False

        try:

            if self.cap is not None:

                self.cap.release()

        except Exception:

            pass

        try:

            self.root.destroy()

        except Exception:

            pass


# ============================================================
# main
# ============================================================

def main():

    print("=" * 70)

    print(
        "人流进出计数与区域热度分析"
    )

    print(
        "Realtime Dashboard V13.3"
    )

    print("=" * 70)

    print(
        f"视频：{VIDEO_PATH}"
    )

    print(
        f"模型：{MODEL_PATH}"
    )

    print(
        f"设备：{DEVICE}"
    )

    print("=" * 70)

    root = tk.Tk()

    app = RealtimeDashboardV13_3(
        root
    )

    root.mainloop()


# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":

    main()