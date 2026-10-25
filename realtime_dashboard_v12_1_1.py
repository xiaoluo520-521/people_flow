import cv2
import os
import time
import threading
import queue
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from ultralytics import YOLO


# ============================================================
# 1. 路径配置
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

import json

if os.path.exists(CONFIG_PATH):

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    line_cfg = config.get("counting_line", {})

    P1 = tuple(line_cfg.get("p1", [200, 360]))
    P2 = tuple(line_cfg.get("p2", [1100, 360]))

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
# 3. 实时 Dashboard
# ============================================================

class RealtimeDashboard:

    def __init__(self, root):

        self.root = root
        self.root.title("人流进出计数与区域热度分析 - 实时监控")
        self.root.geometry("1500x900")
        self.root.minsize(1200, 750)

        # ----------------------------------------------------
        # 视频
        # ----------------------------------------------------

        self.cap = cv2.VideoCapture(VIDEO_PATH)

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

        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        print("正在加载 YOLO 模型...")

        self.model = YOLO(MODEL_PATH)

        print("YOLO 模型加载完成")

        # ----------------------------------------------------
        # 播放状态
        # ----------------------------------------------------

        self.running = False
        self.paused = False
        self.stop_flag = False

        self.worker_thread = None

        # 最新视频帧
        self.frame_queue = queue.Queue(maxsize=1)

        # ----------------------------------------------------
        # 实时统计
        # ----------------------------------------------------

        self.current_people = 0

        self.in_count = 0
        self.out_count = 0

        self.track_ids = set()

        # ----------------------------------------------------
        # ROI统计
        # ----------------------------------------------------

        # 所有曾经进入过ROI的人
        self.roi_visitors = set()

        # 当前正在ROI里面的人
        self.roi_inside = set()

        # ROI总进入次数
        self.roi_visits = 0

        # ----------------------------------------------------
        # 计数线状态
        # ----------------------------------------------------

        self.previous_side = {}

        self.counted_ids = set()

        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        self.last_time = time.time()
        self.frame_counter = 0
        self.current_fps = 0

        # ----------------------------------------------------
        # UI
        # ----------------------------------------------------

        self.create_ui()

        # ----------------------------------------------------
        # 首帧
        # ----------------------------------------------------

        self.show_first_frame()

        # ----------------------------------------------------
        # 启动UI刷新
        # ----------------------------------------------------

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

        # 主标题
        title = tk.Label(
            self.root,
            text="人流进出计数与区域热度分析",
            font=("Microsoft YaHei", 22, "bold")
        )

        title.pack(
            pady=10
        )

        # ----------------------------------------------------
        # 主区域
        # ----------------------------------------------------

        main_frame = tk.Frame(
            self.root
        )

        main_frame.pack(
            fill=tk.BOTH,
            expand=True,
            padx=10
        )

        # ====================================================
        # 左边视频
        # ====================================================

        left_frame = tk.Frame(
            main_frame
        )

        left_frame.pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True
        )

        video_title = tk.Label(
            left_frame,
            text="实时视频",
            font=("Microsoft YaHei", 15, "bold")
        )

        video_title.pack(
            pady=5
        )

        self.video_label = tk.Label(
            left_frame,
            bg="black",
            width=1000,
            height=600
        )

        self.video_label.pack(
            fill=tk.BOTH,
            expand=True
        )

        # ====================================================
        # 右边数据
        # ====================================================

        right_frame = tk.Frame(
            main_frame,
            width=350
        )

        right_frame.pack(
            side=tk.RIGHT,
            fill=tk.Y,
            padx=(15, 0)
        )

        right_frame.pack_propagate(False)

        data_title = tk.Label(
            right_frame,
            text="实时数据",
            font=("Microsoft YaHei", 18, "bold")
        )

        data_title.pack(
            pady=10
        )

        # ----------------------------------------------------
        # KPI
        # ----------------------------------------------------

        self.kpi_labels = {}

        kpis = [
            ("current", "当前人数"),
            ("in", "进入人数"),
            ("out", "离开人数"),
            ("total", "累计进出"),
            ("roi_current", "ROI当前人数"),
            ("roi_visitors", "ROI访问人数"),
            ("roi_visits", "ROI访问次数"),
            ("global_ids", "当前Track ID"),
            ("fps", "实时FPS")
        ]

        for key, name in kpis:

            frame = tk.Frame(
                right_frame,
                bd=1,
                relief=tk.GROOVE
            )

            frame.pack(
                fill=tk.X,
                pady=4
            )

            name_label = tk.Label(
                frame,
                text=name,
                font=("Microsoft YaHei", 11)
            )

            name_label.pack(
                side=tk.LEFT,
                padx=10,
                pady=8
            )

            value_label = tk.Label(
                frame,
                text="0",
                font=("Microsoft YaHei", 14, "bold")
            )

            value_label.pack(
                side=tk.RIGHT,
                padx=10
            )

            self.kpi_labels[key] = value_label

        # ----------------------------------------------------
        # 视频信息
        # ----------------------------------------------------

        info_frame = tk.LabelFrame(
            right_frame,
            text="视频信息",
            font=("Microsoft YaHei", 11)
        )

        info_frame.pack(
            fill=tk.X,
            pady=15
        )

        self.info_label = tk.Label(
            info_frame,
            text="",
            justify=tk.LEFT,
            anchor="w",
            font=("Microsoft YaHei", 10)
        )

        self.info_label.pack(
            padx=10,
            pady=10
        )

        # ====================================================
        # 控制区域
        # ====================================================

        control_frame = tk.Frame(
            self.root
        )

        control_frame.pack(
            fill=tk.X,
            padx=10,
            pady=8
        )

        # ----------------------------------------------------
        # 播放按钮
        # ----------------------------------------------------

        self.play_button = tk.Button(
            control_frame,
            text="▶ 播放",
            font=("Microsoft YaHei", 11),
            width=10,
            command=self.play_video
        )

        self.play_button.pack(
            side=tk.LEFT,
            padx=5
        )

        # ----------------------------------------------------

        self.pause_button = tk.Button(
            control_frame,
            text="⏸ 暂停",
            font=("Microsoft YaHei", 11),
            width=10,
            command=self.pause_video
        )

        self.pause_button.pack(
            side=tk.LEFT,
            padx=5
        )

        # ----------------------------------------------------

        self.restart_button = tk.Button(
            control_frame,
            text="↻ 重播",
            font=("Microsoft YaHei", 11),
            width=10,
            command=self.restart_video
        )

        self.restart_button.pack(
            side=tk.LEFT,
            padx=5
        )

        # ----------------------------------------------------

        self.stop_button = tk.Button(
            control_frame,
            text="■ 停止",
            font=("Microsoft YaHei", 11),
            width=10,
            command=self.stop_video
        )

        self.stop_button.pack(
            side=tk.LEFT,
            padx=5
        )

        # ----------------------------------------------------
        # 状态
        # ----------------------------------------------------

        self.status_label = tk.Label(
            control_frame,
            text="状态：已准备",
            font=("Microsoft YaHei", 11)
        )

        self.status_label.pack(
            side=tk.LEFT,
            padx=20
        )

        # ====================================================
        # 进度条
        # ====================================================

        progress_frame = tk.Frame(
            self.root
        )

        progress_frame.pack(
            fill=tk.X,
            padx=15,
            pady=(0, 10)
        )

        self.progress = ttk.Progressbar(
            progress_frame,
            orient="horizontal",
            mode="determinate"
        )

        self.progress.pack(
            fill=tk.X
        )

    # ========================================================
    # 显示首帧
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

        time.sleep(0.05)

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

        self.frame_counter = 0
        self.current_fps = 0

        # ByteTrack重新开始
        try:
            self.model.predictor = None
        except Exception:
            pass

        self.show_first_frame()

        self.status_label.config(
            text="状态：已重置"
        )

    # ========================================================
    # 视频处理线程
    # ========================================================

    def video_worker(self):

        last_frame_time = time.time()

        while self.running and not self.stop_flag:

            if self.paused:

                time.sleep(0.05)
                continue

            ret, frame = self.cap.read()

            if not ret:

                self.running = False

                self.root.after(
                    0,
                    lambda: self.status_label.config(
                        text="状态：播放结束"
                    )
                )

                break

            # ------------------------------------------------
            # YOLO + ByteTrack
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

            current_ids = []

            if result.boxes is not None:

                boxes = result.boxes

                if boxes.id is not None:

                    ids = (
                        boxes.id
                        .cpu()
                        .numpy()
                        .astype(int)
                    )

                    xyxy = (
                        boxes.xyxy
                        .cpu()
                        .numpy()
                    )

                    current_ids = ids.tolist()

                    self.current_people = len(
                        current_ids
                    )

                    self.track_ids = set(
                        current_ids
                    )

                    # ========================================
                    # 处理每个人
                    # ========================================

                    for track_id, box in zip(
                        ids,
                        xyxy
                    ):

                        track_id = int(track_id)

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

                        # ------------------------------------
                        # 计数线
                        # ------------------------------------

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

                            # 上方 -> 下方
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

                            # 下方 -> 上方
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

                        # ------------------------------------
                        # ROI
                        # ------------------------------------

                        inside = self.point_in_roi(
                            point
                        )

                        if inside:

                            # 历史访问人数
                            self.roi_visitors.add(
                                track_id
                            )

                            # 第一次进入当前ROI
                            if (
                                track_id
                                not in self.roi_inside
                            ):

                                self.roi_inside.add(
                                    track_id
                                )

                                self.roi_visits += 1

                        else:

                            # 离开ROI
                            self.roi_inside.discard(
                                track_id
                            )

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

            self.frame_counter += 1

            now = time.time()

            if now - last_frame_time >= 1.0:

                self.current_fps = (
                    self.frame_counter
                    /
                    (now - last_frame_time)
                )

                self.frame_counter = 0

                last_frame_time = now

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

        self.kpi_labels[
            "current"
        ].config(
            text=str(
                self.current_people
            )
        )

        self.kpi_labels[
            "in"
        ].config(
            text=str(
                self.in_count
            )
        )

        self.kpi_labels[
            "out"
        ].config(
            text=str(
                self.out_count
            )
        )

        self.kpi_labels[
            "total"
        ].config(
            text=str(
                self.in_count
                +
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
            "global_ids"
        ].config(
            text=str(
                len(
                    self.track_ids
                )
            )
        )

        self.kpi_labels[
            "fps"
        ].config(
            text=f"{self.current_fps:.1f}"
        )

        # ----------------------------------------------------
        # 视频信息
        # ----------------------------------------------------

        current_frame = int(
            self.cap.get(
                cv2.CAP_PROP_POS_FRAMES
            )
        )

        progress = 0

        if self.total_frames > 0:

            progress = (
                current_frame
                /
                self.total_frames
                *
                100
            )

        self.progress["value"] = progress

        current_time = (
            current_frame
            /
            self.fps_video
        )

        total_time = (
            self.total_frames
            /
            self.fps_video
        )

        self.info_label.config(
            text=(
                f"分辨率："
                f"{self.video_width} × "
                f"{self.video_height}\n"
                f"FPS："
                f"{self.fps_video:.2f}\n"
                f"总帧数："
                f"{self.total_frames}\n"
                f"当前帧："
                f"{current_frame}\n"
                f"播放时间："
                f"{current_time:.1f}s / "
                f"{total_time:.1f}s"
            )
        )

        self.root.after(
            30,
            self.update_ui
        )

    # ========================================================
    # 显示视频
    # ========================================================

    def display_frame(
        self,
        frame
    ):

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(
            frame_rgb
        )

        # 控制最大显示尺寸
        max_width = 950
        max_height = 650

        ratio = min(
            max_width / image.width,
            max_height / image.height
        )

        new_size = (
            int(image.width * ratio),
            int(image.height * ratio)
        )

        image = image.resize(
            new_size,
            Image.Resampling.LANCZOS
        )

        photo = ImageTk.PhotoImage(
            image=image
        )

        self.video_label.config(
            image=photo
        )

        self.video_label.image = photo

    # ========================================================
    # 绘制
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

            pts = (
                __import__("numpy")
                .array(
                    ROI_POINTS,
                    dtype="int32"
                )
            )

            cv2.polylines(
                frame,
                [pts],
                True,
                (255, 0, 0),
                3
            )

            # ROI半透明区域

            overlay = frame.copy()

            cv2.fillPoly(
                overlay,
                [pts],
                (255, 0, 0)
            )

            frame = cv2.addWeighted(
                overlay,
                0.15,
                frame,
                0.85,
                0
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
        # YOLO检测框
        # ----------------------------------------------------

        if result is not None:

            if (
                result.boxes is not None
                and
                result.boxes.id is not None
            ):

                boxes = result.boxes

                ids = (
                    boxes.id
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                xyxy = (
                    boxes.xyxy
                    .cpu()
                    .numpy()
                )

                for track_id, box in zip(
                    ids,
                    xyxy
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
                        (x1, max(20, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

        # ----------------------------------------------------
        # 左上角实时数据
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (10, 10),
            (320, 160),
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

        for i, txt in enumerate(texts):

            cv2.putText(
                frame,
                txt,
                (20, 38 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

        return frame

    # ========================================================
    # 计数线方向
    # ========================================================

    def get_side(
        self,
        point
    ):

        x, y = point

        x1, y1 = P1
        x2, y2 = P2

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

    # ========================================================
    # ROI判断
    # ========================================================

    def point_in_roi(
        self,
        point
    ):

        if len(ROI_POINTS) < 3:

            return False

        pts = (
            __import__("numpy")
            .array(
                ROI_POINTS,
                dtype="int32"
            )
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

        time.sleep(0.1)

        if self.cap is not None:

            self.cap.release()

        self.root.destroy()


# ============================================================
# 4. 主程序
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("人流进出计数与区域热度分析 - 实时 Dashboard")
    print("=" * 60)

    print(f"视频：{VIDEO_PATH}")
    print(f"模型：{MODEL_PATH}")

    if not os.path.exists(VIDEO_PATH):

        print("❌ 视频不存在")
        exit()

    if not os.path.exists(MODEL_PATH):

        print("❌ YOLO模型不存在")
        print(
            f"请检查：{MODEL_PATH}"
        )
        exit()

    root = tk.Tk()

    app = RealtimeDashboard(
        root
    )

    root.mainloop()