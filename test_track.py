from ultralytics import YOLO
import os

model = YOLO("yolo11n.pt")

video_path = "videos/test.mp4"
output_dir = "results"

os.makedirs(output_dir, exist_ok=True)

results = model.track(
    source=video_path,
    tracker="bytetrack.yaml",
    persist=True,
    classes=[0],
    conf=0.3,
    device="cuda:0",
    save=True,
    project=output_dir,
    name="track_gpu",
    show=False,
    verbose=True
)

print("\n==============================")
print("GPU 视频跟踪完成！")
print("==============================")