import cv2

VIDEO_PATH = r"E:\people_flow\videos\test.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ 视频打开失败")
    exit()

print("✅ 视频打开成功")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"FPS：{fps}")
print(f"分辨率：{width} × {height}")
print(f"总帧数：{frames}")

while True:

    ret, frame = cap.read()

    if not ret:
        print("视频播放结束")
        break

    cv2.imshow(
        "Video Test",
        frame
    )

    key = cv2.waitKey(30) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()