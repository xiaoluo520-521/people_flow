from ultralytics import YOLO
import cv2
import csv
import os


# ============================================================
# 1. 基本配置
# ============================================================

MODEL_PATH = r"E:\people_flow\yolo11n.pt"
VIDEO_PATH = r"E:\people_flow\videos\test.mp4"

OUTPUT_DIR = r"E:\people_flow\results\trajectory"

CSV_PATH = os.path.join(OUTPUT_DIR, "tracks.csv")
VIDEO_OUTPUT = os.path.join(OUTPUT_DIR, "trajectory.mp4")


# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. 加载模型
# ============================================================

print("正在加载 YOLO 模型...")

model = YOLO(MODEL_PATH)

print("模型加载成功！")


# ============================================================
# 3. 打开视频
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"无法打开视频：{VIDEO_PATH}")


fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("\n==============================")
print("视频信息")
print("==============================")
print(f"分辨率：{width} × {height}")
print(f"FPS：{fps:.2f}")
print(f"总帧数：{total_frames}")
print(f"时长：{total_frames / fps:.2f} 秒")


# ============================================================
# 4. 创建输出视频
# ============================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    VIDEO_OUTPUT,
    fourcc,
    fps,
    (width, height)
)


# ============================================================
# 5. 创建 CSV
# ============================================================

csv_file = open(
    CSV_PATH,
    mode="w",
    newline="",
    encoding="utf-8-sig"
)

csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "frame",
    "time",
    "track_id",
    "cx",
    "cy",
    "width",
    "height"
])


# ============================================================
# 6. 保存轨迹
# ============================================================

# 用字典保存每个人的历史轨迹
trajectories = {}

frame_id = 0


print("\n开始进行目标检测与跟踪...")
print("--------------------------------")


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_id += 1

    # ========================================================
    # YOLO + ByteTrack
    # ========================================================

    results = model.track(
        frame,
        tracker="bytetrack.yaml",
        persist=True,
        classes=[0],
        conf=0.3,
        verbose=False
    )

    result = results[0]

    # ========================================================
    # 判断当前帧有没有检测结果
    # ========================================================

    if result.boxes is not None and result.boxes.id is not None:

        boxes = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.int().cpu().numpy()

        # ====================================================
        # 遍历当前帧所有行人
        # ====================================================

        for box, track_id in zip(boxes, track_ids):

            x1, y1, x2, y2 = box

            # 中心点
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # 框尺寸
            box_width = x2 - x1
            box_height = y2 - y1

            # 时间
            current_time = frame_id / fps

            # =================================================
            # 写入 CSV
            # =================================================

            csv_writer.writerow([
                frame_id,
                round(current_time, 3),
                int(track_id),
                round(cx, 2),
                round(cy, 2),
                round(box_width, 2),
                round(box_height, 2)
            ])

            # =================================================
            # 保存历史轨迹
            # =================================================

            if int(track_id) not in trajectories:

                trajectories[int(track_id)] = []

            trajectories[int(track_id)].append(
                (int(cx), int(cy))
            )

            # =================================================
            # 绘制检测框
            # =================================================

            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )

            # =================================================
            # 绘制 ID
            # =================================================

            label = f"Person ID: {int(track_id)}"

            cv2.putText(
                frame,
                label,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )


    # ========================================================
    # 绘制历史轨迹
    # ========================================================

    for track_id, points in trajectories.items():

        # 只绘制最近 50 个点
        recent_points = points[-50:]

        for i in range(1, len(recent_points)):

            pt1 = recent_points[i - 1]
            pt2 = recent_points[i]

            cv2.line(
                frame,
                pt1,
                pt2,
                (255, 0, 0),
                2
            )


    # ========================================================
    # 显示当前帧编号
    # ========================================================

    cv2.putText(
        frame,
        f"Frame: {frame_id}/{total_frames}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )


    # ========================================================
    # 写入输出视频
    # ========================================================

    writer.write(frame)


    # ========================================================
    # 终端显示进度
    # ========================================================

    if frame_id % 50 == 0:

        progress = frame_id / total_frames * 100

        print(
            f"处理进度：{progress:.1f}% "
            f"({frame_id}/{total_frames})"
        )


# ============================================================
# 7. 释放资源
# ============================================================

cap.release()
writer.release()
csv_file.close()


# ============================================================
# 8. 输出统计信息
# ============================================================

print("\n====================================")
print("轨迹提取完成！")
print("====================================")

print(f"总帧数：{frame_id}")

print(f"检测到的不同 ID 数量：{len(trajectories)}")

print("\n所有 ID：")

for track_id in sorted(trajectories.keys()):

    print(
        f"ID {track_id}: "
        f"{len(trajectories[track_id])} 个轨迹点"
    )

print("\nCSV 文件：")
print(CSV_PATH)

print("\n轨迹视频：")
print(VIDEO_OUTPUT)

print("\n====================================")