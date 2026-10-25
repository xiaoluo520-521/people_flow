import cv2
import os
import pandas as pd
import numpy as np


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

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "region_dwell_v6"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# 输出文件
EVENT_CSV = os.path.join(
    OUTPUT_DIR,
    "region_dwell_events_v6.csv"
)

SUMMARY_CSV = os.path.join(
    OUTPUT_DIR,
    "region_dwell_summary_v6.csv"
)

OUTPUT_VIDEO = os.path.join(
    OUTPUT_DIR,
    "region_dwell_result_v6.mp4"
)


# ============================================================
# 2. 参数
# ============================================================

# 最少轨迹长度
MIN_TRACK_FRAMES = 10

# 最大允许的轨迹帧间隔
# 如果中间丢了太多帧，不把两点之间的时间直接算作连续停留
MAX_FRAME_GAP = 2

# ROI显示透明度
ROI_ALPHA = 0.25


# ============================================================
# 3. 全局变量：鼠标绘制 ROI
# ============================================================

roi_points = []
drawing_finished = False


def mouse_callback(event, x, y, flags, param):
    global roi_points

    if event == cv2.EVENT_LBUTTONDOWN:

        if not drawing_finished:
            roi_points.append((x, y))

            print(
                f"添加 ROI 点：({x}, {y})"
            )


# ============================================================
# 4. 绘制 ROI
# ============================================================

def select_roi(video_path):

    global roi_points
    global drawing_finished

    roi_points = []
    drawing_finished = False

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"无法打开视频：{video_path}"
        )

    ret, frame = cap.read()

    if not ret:
        cap.release()
        raise RuntimeError(
            "无法读取视频第一帧"
        )

    cap.release()

    window_name = "ROI Selection"

    cv2.namedWindow(window_name)

    cv2.setMouseCallback(
        window_name,
        mouse_callback
    )

    print("\n======================================")
    print("ROI 区域绘制")
    print("======================================")
    print("鼠标左键：添加区域顶点")
    print("R：重新绘制")
    print("Enter：确认")
    print("Esc：退出")
    print("======================================\n")

    while True:

        display = frame.copy()

        # ----------------------------------------------------
        # 绘制已经选择的点
        # ----------------------------------------------------

        for i, point in enumerate(roi_points):

            cv2.circle(
                display,
                point,
                6,
                (0, 255, 0),
                -1
            )

            cv2.putText(
                display,
                str(i + 1),
                (
                    point[0] + 8,
                    point[1] - 8
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # ----------------------------------------------------
        # 绘制多边形
        # ----------------------------------------------------

        if len(roi_points) >= 2:

            pts = np.array(
                roi_points,
                dtype=np.int32
            )

            cv2.polylines(
                display,
                [pts],
                False,
                (0, 255, 0),
                2
            )

        if len(roi_points) >= 3:

            overlay = display.copy()

            pts = np.array(
                roi_points,
                dtype=np.int32
            )

            cv2.fillPoly(
                overlay,
                [pts],
                (0, 255, 0)
            )

            display = cv2.addWeighted(
                overlay,
                ROI_ALPHA,
                display,
                1 - ROI_ALPHA,
                0
            )

            cv2.polylines(
                display,
                [pts],
                True,
                (0, 255, 0),
                3
            )

        # ----------------------------------------------------
        # 提示信息
        # ----------------------------------------------------

        cv2.rectangle(
            display,
            (10, 10),
            (500, 80),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            display,
            "Left Click: Add Point | R: Reset | Enter: Confirm",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

        cv2.putText(
            display,
            f"Points: {len(roi_points)}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

        cv2.imshow(
            window_name,
            display
        )

        key = cv2.waitKey(30) & 0xFF

        # Enter
        if key == 13:

            if len(roi_points) >= 3:

                drawing_finished = True

                print("\nROI 区域确认：")
                print(roi_points)

                break

            else:

                print(
                    "至少需要 3 个点才能构成多边形！"
                )

        # R
        elif key == ord("r"):

            roi_points = []

            print(
                "ROI 已清除，请重新绘制。"
            )

        # ESC
        elif key == 27:

            cv2.destroyAllWindows()

            raise SystemExit(
                "用户取消 ROI 绘制。"
            )

    cv2.destroyWindow(
        window_name
    )

    return roi_points


# ============================================================
# 5. 判断点是否在 ROI 内
# ============================================================

def point_inside_roi(
    x,
    y,
    polygon
):

    polygon_np = np.array(
        polygon,
        dtype=np.int32
    )

    result = cv2.pointPolygonTest(
        polygon_np,
        (float(x), float(y)),
        False
    )

    return result >= 0


# ============================================================
# 6. 计算区域停留时间
# ============================================================

def calculate_dwell(
    df,
    polygon
):

    print("\n======================================")
    print("开始计算区域停留时间")
    print("======================================")

    events = []

    summary = []

    # --------------------------------------------------------
    # Global ID 分组
    # --------------------------------------------------------

    grouped = df.groupby(
        "global_id"
    )

    for global_id, group in grouped:

        group = group.sort_values(
            "frame"
        ).reset_index(drop=True)

        if len(group) < MIN_TRACK_FRAMES:
            continue

        # 当前是否在区域内
        inside_list = []

        for _, row in group.iterrows():

            inside = point_inside_roi(
                row["cx_smooth"],
                row["cy_smooth"],
                polygon
            )

            inside_list.append(
                inside
            )

        group["inside"] = inside_list

        # ----------------------------------------------------
        # 状态变量
        # ----------------------------------------------------

        in_region = False

        enter_frame = None
        enter_time = None

        total_dwell = 0.0

        visit_count = 0

        max_dwell = 0.0

        # ----------------------------------------------------
        # 遍历轨迹
        # ----------------------------------------------------

        for i in range(len(group)):

            row = group.iloc[i]

            current_inside = row["inside"]

            current_frame = int(
                row["frame"]
            )

            current_time = float(
                row["time"]
            )

            # =================================================
            # 区域外 → 区域内
            # =================================================

            if current_inside and not in_region:

                in_region = True

                enter_frame = current_frame
                enter_time = current_time

                visit_count += 1

            # =================================================
            # 区域内
            # =================================================

            elif current_inside and in_region:

                prev_row = group.iloc[i - 1]

                prev_frame = int(
                    prev_row["frame"]
                )

                # 正常连续轨迹
                if current_frame - prev_frame <= MAX_FRAME_GAP:

                    pass

                # 如果中间丢帧太多
                else:

                    # 先结束上一段
                    exit_frame = int(
                        prev_row["frame"]
                    )

                    exit_time = float(
                        prev_row["time"]
                    )

                    dwell_time = (
                        exit_time -
                        enter_time
                    )

                    if dwell_time < 0:
                        dwell_time = 0

                    events.append({
                        "global_id": int(global_id),
                        "visit_id": visit_count,
                        "enter_frame": enter_frame,
                        "enter_time": enter_time,
                        "exit_frame": exit_frame,
                        "exit_time": exit_time,
                        "dwell_time": dwell_time
                    })

                    total_dwell += dwell_time

                    max_dwell = max(
                        max_dwell,
                        dwell_time
                    )

                    # 重新开始
                    enter_frame = current_frame
                    enter_time = current_time

            # =================================================
            # 区域内 → 区域外
            # =================================================

            elif not current_inside and in_region:

                prev_row = group.iloc[i - 1]

                exit_frame = int(
                    prev_row["frame"]
                )

                exit_time = float(
                    prev_row["time"]
                )

                dwell_time = (
                    exit_time -
                    enter_time
                )

                if dwell_time < 0:
                    dwell_time = 0

                events.append({
                    "global_id": int(global_id),
                    "visit_id": visit_count,
                    "enter_frame": enter_frame,
                    "enter_time": enter_time,
                    "exit_frame": exit_frame,
                    "exit_time": exit_time,
                    "dwell_time": dwell_time
                })

                total_dwell += dwell_time

                max_dwell = max(
                    max_dwell,
                    dwell_time
                )

                in_region = False

                enter_frame = None
                enter_time = None

        # ----------------------------------------------------
        # 视频结束时还在区域内
        # ----------------------------------------------------

        if in_region:

            last_row = group.iloc[-1]

            exit_frame = int(
                last_row["frame"]
            )

            exit_time = float(
                last_row["time"]
            )

            dwell_time = (
                exit_time -
                enter_time
            )

            if dwell_time < 0:
                dwell_time = 0

            events.append({
                "global_id": int(global_id),
                "visit_id": visit_count,
                "enter_frame": enter_frame,
                "enter_time": enter_time,
                "exit_frame": exit_frame,
                "exit_time": exit_time,
                "dwell_time": dwell_time
            })

            total_dwell += dwell_time

            max_dwell = max(
                max_dwell,
                dwell_time
            )

        # ----------------------------------------------------
        # 汇总
        # ----------------------------------------------------

        if visit_count > 0:

            average_dwell = (
                total_dwell /
                visit_count
            )

        else:

            average_dwell = 0

        summary.append({
            "global_id": int(global_id),
            "visit_count": visit_count,
            "total_dwell_time": total_dwell,
            "average_dwell_time": average_dwell,
            "max_dwell_time": max_dwell
        })

    # ========================================================
    # 保存结果
    # ========================================================

    events_df = pd.DataFrame(
        events
    )

    summary_df = pd.DataFrame(
        summary
    )

    events_df.to_csv(
        EVENT_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    summary_df.to_csv(
        SUMMARY_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n区域停留统计完成")

    print(
        f"区域访问记录：{len(events_df)}"
    )

    print(
        f"涉及 Global ID：{len(summary_df)}"
    )

    if len(events_df) > 0:

        print(
            f"总停留时间："
            f"{events_df['dwell_time'].sum():.2f} s"
        )

        print(
            f"平均单次停留："
            f"{events_df['dwell_time'].mean():.2f} s"
        )

    print(
        f"\n详细记录：{EVENT_CSV}"
    )

    print(
        f"汇总结果：{SUMMARY_CSV}"
    )

    return events_df, summary_df


# ============================================================
# 7. 生成可视化视频
# ============================================================

def create_visualization(
    video_path,
    track_df,
    polygon,
    events_df,
    summary_df
):

    print("\n======================================")
    print("开始生成区域停留可视化视频")
    print("======================================")

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        raise RuntimeError(
            "无法打开视频"
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

    if fps <= 0:
        fps = 30

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
    # 建立 frame -> tracks
    # --------------------------------------------------------

    frame_groups = {
        frame: group
        for frame, group
        in track_df.groupby("frame")
    }

    # --------------------------------------------------------
    # 建立 Global ID -> 停留统计
    # --------------------------------------------------------

    summary_dict = {}

    for _, row in summary_df.iterrows():

        summary_dict[
            int(row["global_id"])
        ] = row

    # --------------------------------------------------------
    # 视频处理
    # --------------------------------------------------------

    frame_idx = 0

    cumulative_visitors = set()

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_idx += 1

        display = frame.copy()

        # ====================================================
        # ROI
        # ====================================================

        polygon_np = np.array(
            polygon,
            dtype=np.int32
        )

        overlay = display.copy()

        cv2.fillPoly(
            overlay,
            [polygon_np],
            (0, 255, 0)
        )

        display = cv2.addWeighted(
            overlay,
            ROI_ALPHA,
            display,
            1 - ROI_ALPHA,
            0
        )

        cv2.polylines(
            display,
            [polygon_np],
            True,
            (0, 255, 0),
            3
        )

        # ====================================================
        # 当前帧轨迹
        # ====================================================

        current_people = 0

        if frame_idx in frame_groups:

            current = frame_groups[
                frame_idx
            ]

            for _, row in current.iterrows():

                gid = int(
                    row["global_id"]
                )

                x = int(
                    row["cx_smooth"]
                )

                y = int(
                    row["cy_smooth"]
                )

                inside = point_inside_roi(
                    x,
                    y,
                    polygon
                )

                if inside:

                    current_people += 1

                    cumulative_visitors.add(
                        gid
                    )

                    point_color = (
                        0,
                        0,
                        255
                    )

                else:

                    point_color = (
                        255,
                        255,
                        0
                    )

                # ------------------------------------------------
                # 中心点
                # ------------------------------------------------

                cv2.circle(
                    display,
                    (x, y),
                    5,
                    point_color,
                    -1
                )

                # ------------------------------------------------
                # Global ID
                # ------------------------------------------------

                label = f"GID {gid}"

                cv2.putText(
                    display,
                    label,
                    (
                        x + 8,
                        y - 8
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    point_color,
                    2
                )

        # ====================================================
        # 当前平均停留时间
        # ====================================================

        average_dwell = 0

        if len(events_df) > 0:

            completed_events = events_df[
                events_df["exit_frame"] <= frame_idx
            ]

            if len(completed_events) > 0:

                average_dwell = (
                    completed_events[
                        "dwell_time"
                    ].mean()
                )

        # ====================================================
        # 信息面板
        # ====================================================

        cv2.rectangle(
            display,
            (10, 10),
            (390, 155),
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
            f"People in ROI: {current_people}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )

        cv2.putText(
            display,
            f"Unique Visitors: {len(cumulative_visitors)}",
            (20, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )

        cv2.putText(
            display,
            f"Visits: {len(events_df)}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )

        cv2.putText(
            display,
            f"Avg Dwell: {average_dwell:.2f}s",
            (20, 135),
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
        # 显示处理进度
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

    print("\n======================================")
    print("视频生成完成")
    print("======================================")

    print(
        f"输出视频：{OUTPUT_VIDEO}"
    )


# ============================================================
# 8. 主程序
# ============================================================

def main():

    print("\n")
    print("==============================================")
    print("        区域停留时间分析 V6")
    print("==============================================")

    # --------------------------------------------------------
    # 检查文件
    # --------------------------------------------------------

    if not os.path.exists(
        VIDEO_PATH
    ):

        raise FileNotFoundError(
            f"视频不存在：{VIDEO_PATH}"
        )

    if not os.path.exists(
        TRACK_CSV
    ):

        raise FileNotFoundError(
            f"轨迹文件不存在：{TRACK_CSV}"
        )

    # --------------------------------------------------------
    # 读取轨迹
    # --------------------------------------------------------

    print(
        f"\n读取轨迹：{TRACK_CSV}"
    )

    df = pd.read_csv(
        TRACK_CSV
    )

    print(
        f"轨迹数据量：{len(df)}"
    )

    print(
        f"Global ID 数量："
        f"{df['global_id'].nunique()}"
    )

    # --------------------------------------------------------
    # 过滤短轨迹
    # --------------------------------------------------------

    track_counts = (
        df.groupby("global_id")
        .size()
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
        f"过滤后 Global ID："
        f"{df['global_id'].nunique()}"
    )

    # --------------------------------------------------------
    # 检查平滑坐标
    # --------------------------------------------------------

    if (
        "cx_smooth" not in df.columns
        or
        "cy_smooth" not in df.columns
    ):

        raise RuntimeError(
            "轨迹文件中没有 "
            "cx_smooth / cy_smooth"
        )

    # --------------------------------------------------------
    # 手动选择 ROI
    # --------------------------------------------------------

    polygon = select_roi(
        VIDEO_PATH
    )

    print("\n最终 ROI：")

    for i, point in enumerate(
        polygon
    ):

        print(
            f"Point {i + 1}: {point}"
        )

    # --------------------------------------------------------
    # 停留时间计算
    # --------------------------------------------------------

    events_df, summary_df = calculate_dwell(
        df,
        polygon
    )

    # --------------------------------------------------------
    # 生成视频
    # --------------------------------------------------------

    create_visualization(
        VIDEO_PATH,
        df,
        polygon,
        events_df,
        summary_df
    )

    print("\n")
    print("==============================================")
    print("V6 区域停留时间分析完成")
    print("==============================================")

    print(
        f"详细记录：{EVENT_CSV}"
    )

    print(
        f"汇总结果：{SUMMARY_CSV}"
    )

    print(
        f"可视化视频：{OUTPUT_VIDEO}"
    )


if __name__ == "__main__":
    main()