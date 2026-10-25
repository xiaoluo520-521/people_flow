import os
import subprocess
import sys
import time


PROJECT_DIR = r"E:\people_flow"


# ============================================================
# 工具函数
# ============================================================

def run_script(script_name):

    script_path = os.path.join(
        PROJECT_DIR,
        script_name
    )

    print()
    print("=" * 70)
    print(f"开始执行：{script_name}")
    print("=" * 70)

    start_time = time.time()

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=PROJECT_DIR
    )

    elapsed = time.time() - start_time

    print()
    print(
        f"{script_name} 执行时间：{elapsed:.2f} 秒"
    )

    if result.returncode != 0:

        print()
        print(
            f"❌ {script_name} 执行失败"
        )

        return False

    print(
        f"✅ {script_name} 执行成功"
    )

    return True


# ============================================================
# 主流程
# ============================================================

print()
print("=" * 70)
print("        人流进出计数与区域热度分析系统")
print("=" * 70)

print()
print("项目目录：")
print(PROJECT_DIR)


# ============================================================
# 1. 配置
# ============================================================

if not run_script(
    "config_v9_5.py"
):

    sys.exit(1)


# ============================================================
# 2. 轨迹提取
# ============================================================

if not run_script(
    "track_export.py"
):

    sys.exit(1)


# ============================================================
# 3. 轨迹清洗
# ============================================================

if not run_script(
    "trajectory_clean.py"
):

    sys.exit(1)


# ============================================================
# 4. 轨迹重连候选
# ============================================================

if not run_script(
    "trajectory_reconnect_v2.py"
):

    sys.exit(1)


# ============================================================
# 5. ID重连
# ============================================================

if not run_script(
    "trajectory_reconnect_merge_v2.py"
):

    sys.exit(1)


# ============================================================
# 6. 核心分析
# ============================================================

if not run_script(
    "main_v9_4.py"
):

    sys.exit(1)


# ============================================================
# 7. 时间趋势
# ============================================================

if not run_script(
    "time_trend_v11.py"
):

    sys.exit(1)


# ============================================================
# 8. 完成
# ============================================================

print()
print("=" * 70)
print("              全部分析完成")
print("=" * 70)

print()
print("核心结果：")

print(
    r"E:\people_flow\results\unified_v9_4"
)

print()
print("时间趋势：")

print(
    r"E:\people_flow\results\unified_v9_4\time_trend_v11.png"
)

print()
print("轨迹数据：")

print(
    r"E:\people_flow\results\trajectory_reconnect_v2\reconnected_tracks.csv"
)

print()
print("现在可以运行 Dashboard：")

print(
    "python dashboard_v10.py"
)

print("=" * 70)