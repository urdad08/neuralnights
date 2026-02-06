import cv2
import os

INPUT_ROOT = "output_images_complete"
OUTPUT_ROOT = "frames_100x100"
TARGET_SIZE = 100

os.makedirs(OUTPUT_ROOT, exist_ok=True)

for video_folder in sorted(os.listdir(INPUT_ROOT)):
    in_video_dir = os.path.join(INPUT_ROOT, video_folder)

    if not os.path.isdir(in_video_dir):
        continue

    out_video_dir = os.path.join(OUTPUT_ROOT, video_folder)
    os.makedirs(out_video_dir, exist_ok=True)

    for fname in sorted(os.listdir(in_video_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            continue

        in_path = os.path.join(in_video_dir, fname)
        out_path = os.path.join(out_video_dir, fname)

        img = cv2.imread(in_path)
        if img is None:
            print(f"❌ Skipped: {in_path}")
            continue

        h, w = img.shape[:2]
        scale = TARGET_SIZE / min(h, w)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Center crop
        x0 = (new_w - TARGET_SIZE) // 2
        y0 = (new_h - TARGET_SIZE) // 2
        cropped = resized[y0:y0 + TARGET_SIZE, x0:x0 + TARGET_SIZE]

        cv2.imwrite(out_path, cropped)

    print(f"✅ Processed video folder: {video_folder}")

print("🎉 All videos processed successfully")