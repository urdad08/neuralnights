import cv2
import os

INPUT_DIR = "dataset/Train"
OUTPUT_DIR = "dataset_2fps"

os.makedirs(OUTPUT_DIR, exist_ok=True)

VIDEO_EXTS = (".avi", ".wmv", ".webm", ".mp4", ".mov")
TARGET_FPS = 2
FRAME_TIME_MS = 1000 / TARGET_FPS  # 500 ms

for video_name in os.listdir(INPUT_DIR):
    if not video_name.lower().endswith(VIDEO_EXTS):
        continue

    in_path = os.path.join(INPUT_DIR, video_name)
    cap = cv2.VideoCapture(in_path)

    if not cap.isOpened():
        print(f"❌ Cannot open {video_name}")
        continue

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_name = os.path.splitext(video_name)[0] + "_2fps.mp4"
    out_path = os.path.join(OUTPUT_DIR, out_name)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, TARGET_FPS, (width, height))

    next_ts = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ts = cap.get(cv2.CAP_PROP_POS_MSEC)

        if ts >= next_ts:
            out.write(frame)
            next_ts += FRAME_TIME_MS

    cap.release()
    out.release()

    print(f"✅ Done: {video_name}")

print("🎉 All videos converted to 2 FPS")