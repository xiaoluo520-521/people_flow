import os
import subprocess
import sys


# ============================================================
# V10 一键运行系统
# ============================================================

PROJECT_DIR = r"E:\people_flow"

CONFIG_SCRIPT = os.path.join(
    PROJECT_DIR,
    "config_v9_5.py"
)

ANALYSIS_SCRIPT = os.path.join(
    PROJECT_DIR,
    "main_v9_4.py"
)

DASHBOARD_SCRIPT = os.path.join(
    PROJECT_DIR,
    "dashboard_v8.py"
)


def run_script(script_path):

    print()
    print("=" * 70)
    print("正在运行：", os.path.basename(script_path))
    print("=" * 70)

    result = subprocess.run(
        [
            sys.executable,
            script_path
        ],
        cwd=PROJECT_DIR
    )

    if result.returncode != 0:

        print()
        print("程序运行失败：")
        print(script_path)

        return False

    return True


# ============================================================
# 主程序
# ============================================================

print("=" * 70)
print("人流进出计数与区域热度分析系统 V10")
print("=" * 70)

print()
print("系统流程：")
print("1. 可视化设置计数线")
print("2. 可视化设置 ROI")
print("3. 自动进行人流分析")
print("4. 生成进出统计")
print("5. 生成区域停留统计")
print("6. 生成热力图")
print("7. 生成分析视频")
print("8. 启动 Dashboard")
print()


# ============================================================
# 第一步：配置
# ============================================================

print()
print("【1/3】开始设置分析区域...")

if not run_script(CONFIG_SCRIPT):

    print("配置阶段失败，程序结束。")
    sys.exit(1)


# ============================================================
# 第二步：统一分析
# ============================================================

print()
print("【2/3】开始进行视频分析...")

if not run_script(ANALYSIS_SCRIPT):

    print("分析阶段失败，程序结束。")
    sys.exit(1)


# ============================================================
# 第三步：Dashboard
# ============================================================

print()
print("【3/3】启动数据 Dashboard...")

if os.path.exists(DASHBOARD_SCRIPT):

    run_script(DASHBOARD_SCRIPT)

else:

    print()
    print("未找到 Dashboard：")
    print(DASHBOARD_SCRIPT)

    print()
    print("分析已经完成。")


print()
print("=" * 70)
print("V10 系统运行完成")
print("=" * 70)

print()
print("结果目录：")
print(
    os.path.join(
        PROJECT_DIR,
        "results",
        "unified_v9_4"
    )
)

print()