import cv2
import pandas as pd
import os


# ============================================================
# 路径
# ============================================================

VIDEO_PATH = r"E:\people_flow\videos\test.mp4"

CANDIDATE_CSV = (
    r"E:\people_flow\results\trajectory_reconnect_v2"
    r"\reconnect_candidates.csv"
)

OUTPUT_DIR = r"E:\people_flow\results\trajectory_reconnect_v2"

OUTPUT_VIDEO = os.path.join(
    OUTPUT_DIR,
    "candidate_visualization.mp4"
)


# ============================================================
# 参数
# ============================================================

# 每个候选点前后额外显示多少帧
PADDING_FRAMES = 20

# 轨迹显示长度
TRAIL_LENGTH = 30


# ============================================================
# 读取候选
# ============================================================

print("=" * 60)
print("轨迹 ID 重连 V2.1 - 候选可视化")
print("=" * 60)

candidates = pd.read_csv(CANDIDATE_CSV)

print(f"候选数量：{len(candidates)}")


# ============================================================
# 读取视频
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("无法打开视频")


fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

print(f"分辨率：{width} × {height}")
print(f"FPS：{fps:.2f}")
print(f"总帧数：{total_frames}")


# ============================================================
# 视频输出
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)


# ============================================================
# 加载轨迹数据
# ============================================================

TRACK_CSV = (
    r"E:\people_flow\results\trajectory_clean"
    r"\cleaned_tracks.csv"
)

tracks = pd.read_csv(TRACK_CSV)

tracks = tracks.sort_values(
    ["track_id", "frame"]
)


# ============================================================
# 建立 frame -> 轨迹映射
# ============================================================

frame_tracks = {}

for frame_id, group in tracks.groupby("frame"):

    frame_tracks[int(frame_id)] = group


# ============================================================
# 候选逐个生成
# ============================================================

candidate_segments = []

for idx, row in candidates.iterrows():

    id_a = int(row["id_a"])
    id_b = int(row["id_b"])

    end_frame = int(row["end_frame_a"])
    start_frame = int(row["start_frame_b"])

    center_frame = (
        end_frame + start_frame
    ) // 2

    start_frame_display = max(
        1,
        center_frame - PADDING_FRAMES
    )

    end_frame_display = min(
        total_frames,
        center_frame + PADDING_FRAMES
    )

    candidate_segments.append(
        (
            idx + 1,
            id_a,
            id_b,
            end_frame,
            start_frame,
            start_frame_display,
            end_frame_display
        )
    )


# ============================================================
# 输出每个候选
# ============================================================

for (
    candidate_no,
    id_a,
    id_b,
    end_frame_a,
    start_frame_b,
    display_start,
    display_end
) in candidate_segments:

    print()
    print(
        f"[候选 {candidate_no}/{len(candidates)}]"
        f" ID {id_a} -> ID {id_b}"
    )

    # --------------------------------------------------------
    # 获取候选信息
    # --------------------------------------------------------

    row = candidates.iloc[
        candidate_no - 1
    ]

    gap = int(row["gap"])
    distance = float(row["distance"])
    similarity = float(
        row["direction_similarity"]
    )

    # --------------------------------------------------------
    # 视频定位
    # --------------------------------------------------------

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        display_start - 1
    )

    # --------------------------------------------------------
    # 当前候选轨迹历史
    # --------------------------------------------------------

    trail_a = []
    trail_b = []

    current_frame = display_start

    while current_frame <= display_end:

        ret, frame = cap.read()

        if not ret:
            break

        # ====================================================
        # 绘制其他轨迹
        # ====================================================

        if current_frame in frame_tracks:

            frame_data = frame_tracks[
                current_frame
            ]

            for _, p in frame_data.iterrows():

                tid = int(p["track_id"])

                # 只突出当前候选两个ID
                if tid not in [id_a, id_b]:
                    continue

                x = int(p["cx_smooth"])
                y = int(p["cy_smooth"])

                if tid == id_a:

                    trail_a.append(
                        (x, y)
                    )

                    if len(trail_a) > TRAIL_LENGTH:
                        trail_a.pop(0)

                elif tid == id_b:

                    trail_b.append(
                        (x, y)
                    )

                    if len(trail_b) > TRAIL_LENGTH:
                        trail_b.pop(0)

        # ====================================================
        # 绘制 ID A 轨迹
        # ====================================================

        for i in range(
            1,
            len(trail_a)
        ):

            cv2.line(
                frame,
                trail_a[i - 1],
                trail_a[i],
                (0, 255, 255),
                3
            )

        # ====================================================
        # 绘制 ID B 轨迹
        # ====================================================

        for i in range(
            1,
            len(trail_b)
        ):

            cv2.line(
                frame,
                trail_b[i - 1],
                trail_b[i],
                (255, 0, 255),
                3
            )

        # ====================================================
        # 当前点
        # ====================================================

        if current_frame in frame_tracks:

            frame_data = frame_tracks[
                current_frame
            ]

            for _, p in frame_data.iterrows():

                tid = int(p["track_id"])

                if tid not in [id_a, id_b]:
                    continue

                x = int(p["cx_smooth"])
                y = int(p["cy_smooth"])

                # A
                if tid == id_a:

                    cv2.circle(
                        frame,
                        (x, y),
                        8,
                        (0, 255, 255),
                        -1
                    )

                    cv2.putText(
                        frame,
                        f"ID {id_a}",
                        (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2
                    )

                # B
                else:

                    cv2.circle(
                        frame,
                        (x, y),
                        8,
                        (255, 0, 255),
                        -1
                    )

                    cv2.putText(
                        frame,
                        f"ID {id_b}",
                        (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 255),
                        2
                    )

        # ====================================================
        # 候选信息
        # ====================================================

        cv2.rectangle(
            frame,
            (10, 10),
            (570, 125),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            f"Candidate {candidate_no}/{len(candidates)}",
            (20, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"ID {id_a} -> ID {id_b}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Gap: {gap} frames",
            (20, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Distance: {distance:.1f}px",
            (210, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Direction: {similarity:.3f}",
            (390, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Frame: {current_frame}",
            (20, 117),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # ====================================================
        # A/B说明
        # ====================================================

        cv2.putText(
            frame,
            "YELLOW = ID A",
            (width - 250, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "PURPLE = ID B",
            (width - 250, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 0, 255),
            2
        )

        # ====================================================
        # 写入
        # ====================================================

        out.write(frame)

        current_frame += 1


# ============================================================
# 释放
# ============================================================

cap.release()
out.release()

print()
print("=" * 60)
print("可视化完成")
print("=" * 60)
print(f"输出视频：{OUTPUT_VIDEO}")