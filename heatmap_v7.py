import cv2
import os
import pandas as pd
import numpy as np


# ============================================================
# 1. 路径
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

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "heatmap_v7"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


HEATMAP_CSV = os.path.join(
    OUTPUT_DIR,
    "heatmap_data_v7.csv"
)

FINAL_HEATMAP = os.path.join(
    OUTPUT_DIR,
    "heatmap_final_v7.png"
)

OUTPUT_VIDEO = os.path.join(
    OUTPUT_DIR,
    "heatmap_result_v7.mp4"
)


# ============================================================
# 2. 参数
# ============================================================

# 最小轨迹长度
MIN_TRACK_FRAMES = 10

# 热力图扩散半径
BLUR_KERNEL = 51

# 热力图叠加透明度
HEATMAP_ALPHA = 0.45

# 是否绘制 Global ID
SHOW_ID = True

# 是否绘制轨迹
SHOW_TRAIL = True

# 轨迹长度
TRAIL_LENGTH = 30


# ============================================================
# 3. 读取轨迹
# ============================================================

def load_tracks():

    print("\n======================================")
    print("读取 Global ID 轨迹")
    print("======================================")

    if not os.path.exists(
        TRACK_CSV
    ):

        raise FileNotFoundError(
            f"轨迹文件不存在：{TRACK_CSV}"
        )

    df = pd.read_csv(
        TRACK_CSV
    )

    print(
        f"原始轨迹数据：{len(df)}"
    )

    print(
        f"原始 Global ID："
        f"{df['global_id'].nunique()}"
    )

    # --------------------------------------------------------
    # 过滤短轨迹
    # --------------------------------------------------------

    track_counts = (
        df.groupby(
            "global_id"
        ).size()
    )

    valid_ids = track_counts[
        track_counts >= MIN_TRACK_FRAMES
    ].index

    df = df[
        df["global_id"].isin(
            valid_ids
        )
    ].copy()

    print(
        f"过滤后数据：{len(df)}"
    )

    print(
        f"过滤后 Global ID："
        f"{df['global_id'].nunique()}"
    )

    # --------------------------------------------------------
    # 检查坐标
    # --------------------------------------------------------

    if (
        "cx_smooth" not in df.columns
        or
        "cy_smooth" not in df.columns
    ):

        raise RuntimeError(
            "轨迹文件中不存在 "
            "cx_smooth / cy_smooth"
        )

    # --------------------------------------------------------
    # 保存热力图原始数据
    # --------------------------------------------------------

    heatmap_data = pd.DataFrame({

        "frame": df["frame"].astype(int),

        "time": df["time"].astype(float),

        "global_id":
            df["global_id"].astype(int),

        "x":
            df["cx_smooth"].astype(float),

        "y":
            df["cy_smooth"].astype(float)
    })

    heatmap_data.to_csv(
        HEATMAP_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"热力图数据：{HEATMAP_CSV}"
    )

    return df


# ============================================================
# 4. 生成全局热力图
# ============================================================

def generate_final_heatmap(
    df,
    width,
    height
):

    print("\n======================================")
    print("生成最终热力图")
    print("======================================")

    heat = np.zeros(
        (height, width),
        dtype=np.float32
    )

    # --------------------------------------------------------
    # 累积所有轨迹点
    # --------------------------------------------------------

    for _, row in df.iterrows():

        x = int(
            round(
                row["cx_smooth"]
            )
        )

        y = int(
            round(
                row["cy_smooth"]
            )
        )

        if (
            0 <= x < width
            and
            0 <= y < height
        ):

            heat[y, x] += 1.0

    print(
        f"最大原始热度："
        f"{heat.max():.2f}"
    )

    # --------------------------------------------------------
    # 高斯平滑
    # --------------------------------------------------------

    heat_blur = cv2.GaussianBlur(
        heat,
        (
            BLUR_KERNEL,
            BLUR_KERNEL
        ),
        0
    )

    # --------------------------------------------------------
    # 归一化
    # --------------------------------------------------------

    if heat_blur.max() > 0:

        heat_norm = (
            heat_blur /
            heat_blur.max()
        )

    else:

        heat_norm = heat_blur

    heat_uint8 = (
        heat_norm * 255
    ).astype(
        np.uint8
    )

    # --------------------------------------------------------
    # 伪彩色
    # --------------------------------------------------------

    heat_color = cv2.applyColorMap(
        heat_uint8,
        cv2.COLORMAP_JET
    )

    # --------------------------------------------------------
    # 背景
    # --------------------------------------------------------

    background = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    # 使用较浅背景
    background[:] = (
        40,
        40,
        40
    )

    # --------------------------------------------------------
    # 叠加
    # --------------------------------------------------------

    result = cv2.addWeighted(
        background,
        0.35,
        heat_color,
        0.65,
        0
    )

    # --------------------------------------------------------
    # 添加标题
    # --------------------------------------------------------

    cv2.rectangle(
        result,
        (10, 10),
        (330, 55),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        result,
        "Global Pedestrian Heatmap",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------

    cv2.imwrite(
        FINAL_HEATMAP,
        result
    )

    print(
        f"最终热力图：{FINAL_HEATMAP}"
    )

    return heat_norm


# ============================================================
# 5. 动态热力图视频
# ============================================================

def generate_heatmap_video(
    df,
    width,
    height,
    fps,
    total_frames
):

    print("\n======================================")
    print("生成动态热力图视频")
    print("======================================")

    cap = cv2.VideoCapture(
        VIDEO_PATH
    )

    if not cap.isOpened():

        raise RuntimeError(
            "无法打开视频"
        )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        fourcc,
        fps,
        (width, height)
    )

    # --------------------------------------------------------
    # 按 frame 分组
    # --------------------------------------------------------

    frame_groups = {

        frame: group

        for frame, group
        in df.groupby("frame")
    }

    # --------------------------------------------------------
    # 动态热度矩阵
    # --------------------------------------------------------

    heat = np.zeros(
        (height, width),
        dtype=np.float32
    )

    # --------------------------------------------------------
    # 轨迹历史
    # --------------------------------------------------------

    trajectories = {}

    frame_idx = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_idx += 1

        # ====================================================
        # 当前帧轨迹
        # ====================================================

        if frame_idx in frame_groups:

            current = frame_groups[
                frame_idx
            ]

            for _, row in current.iterrows():

                gid = int(
                    row["global_id"]
                )

                x = int(
                    round(
                        row["cx_smooth"]
                    )
                )

                y = int(
                    round(
                        row["cy_smooth"]
                    )
                )

                if (
                    0 <= x < width
                    and
                    0 <= y < height
                ):

                    # ----------------------------------------
                    # 累积热度
                    # ----------------------------------------

                    heat[
                        y,
                        x
                    ] += 1.0

                    # ----------------------------------------
                    # 轨迹历史
                    # ----------------------------------------

                    if gid not in trajectories:

                        trajectories[gid] = []

                    trajectories[gid].append(
                        (x, y)
                    )

                    if len(
                        trajectories[gid]
                    ) > TRAIL_LENGTH:

                        trajectories[gid] = \
                            trajectories[gid][
                                -TRAIL_LENGTH:
                            ]

        # ====================================================
        # 热力图
        # ====================================================

        heat_blur = cv2.GaussianBlur(
            heat,
            (
                BLUR_KERNEL,
                BLUR_KERNEL
            ),
            0
        )

        if heat_blur.max() > 0:

            heat_norm = (
                heat_blur /
                heat_blur.max()
            )

        else:

            heat_norm = heat_blur

        heat_uint8 = (
            heat_norm * 255
        ).astype(
            np.uint8
        )

        heat_color = cv2.applyColorMap(
            heat_uint8,
            cv2.COLORMAP_JET
        )

        # ====================================================
        # 叠加热力图
        # ====================================================

        display = cv2.addWeighted(
            frame,
            1 - HEATMAP_ALPHA,
            heat_color,
            HEATMAP_ALPHA,
            0
        )

        # ====================================================
        # 当前人数
        # ====================================================

        current_people = 0

        if frame_idx in frame_groups:

            current_people = len(
                frame_groups[
                    frame_idx
                ]
            )

        # ====================================================
        # 绘制轨迹
        # ====================================================

        if SHOW_TRAIL:

            for gid, points in trajectories.items():

                if len(points) < 2:
                    continue

                for i in range(
                    1,
                    len(points)
                ):

                    p1 = points[i - 1]
                    p2 = points[i]

                    cv2.line(
                        display,
                        p1,
                        p2,
                        (255, 255, 255),
                        1
                    )

        # ====================================================
        # 绘制当前 Global ID
        # ====================================================

        if (
            SHOW_ID
            and
            frame_idx in frame_groups
        ):

            current = frame_groups[
                frame_idx
            ]

            for _, row in current.iterrows():

                gid = int(
                    row["global_id"]
                )

                x = int(
                    round(
                        row["cx_smooth"]
                    )
                )

                y = int(
                    round(
                        row["cy_smooth"]
                    )
                )

                cv2.circle(
                    display,
                    (x, y),
                    4,
                    (255, 255, 255),
                    -1
                )

                cv2.putText(
                    display,
                    f"GID {gid}",
                    (
                        x + 6,
                        y - 6
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1
                )

        # ====================================================
        # 信息面板
        # ====================================================

        cv2.rectangle(
            display,
            (10, 10),
            (390, 125),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            display,
            f"Frame: {frame_idx}/{total_frames}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )

        cv2.putText(
            display,
            f"Current People: {current_people}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )

        cv2.putText(
            display,
            f"Global IDs: {df['global_id'].nunique()}",
            (20, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )

        cv2.putText(
            display,
            "Heatmap Analysis",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )

        # ====================================================
        # 写入
        # ====================================================

        writer.write(
            display
        )

        # ====================================================
        # 进度
        # ====================================================

        if frame_idx % 100 == 0:

            progress = (
                frame_idx /
                total_frames *
                100
            )

            print(
                f"处理进度："
                f"{frame_idx}/{total_frames}"
                f" ({progress:.1f}%)"
            )

    cap.release()
    writer.release()

    print("\n动态热力图生成完成")

    print(
        f"输出视频：{OUTPUT_VIDEO}"
    )


# ============================================================
# 6. 主程序
# ============================================================

def main():

    print("\n")
    print("==============================================")
    print("          人流热力图分析 V7")
    print("==============================================")

    # --------------------------------------------------------
    # 读取视频信息
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        VIDEO_PATH
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"无法打开视频：{VIDEO_PATH}"
        )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    cap.release()

    if fps <= 0:
        fps = 30

    print(
        f"\n视频分辨率："
        f"{width} × {height}"
    )

    print(
        f"FPS：{fps:.2f}"
    )

    print(
        f"总帧数：{total_frames}"
    )

    print(
        f"视频时长："
        f"{total_frames / fps:.2f}s"
    )

    # --------------------------------------------------------
    # 读取轨迹
    # --------------------------------------------------------

    df = load_tracks()

    # --------------------------------------------------------
    # 最终热力图
    # --------------------------------------------------------

    generate_final_heatmap(
        df,
        width,
        height
    )

    # --------------------------------------------------------
    # 动态视频
    # --------------------------------------------------------

    generate_heatmap_video(
        df,
        width,
        height,
        fps,
        total_frames
    )

    print("\n")
    print("==============================================")
    print("V7 热力图分析完成")
    print("==============================================")

    print(
        f"\n热力图 PNG："
        f"{FINAL_HEATMAP}"
    )

    print(
        f"\n热力图视频："
        f"{OUTPUT_VIDEO}"
    )

    print(
        f"\n热力图数据："
        f"{HEATMAP_CSV}"
    )


if __name__ == "__main__":

    main()