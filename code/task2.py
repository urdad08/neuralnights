import os
import shutil
import pandas as pd

# ===================== PATHS =====================
EXCEL_FILE = "dataset/train_labels.xlsx"          # your Excel file
SOURCE_DIR = "frames_100x100"       # folders with frames
OUTPUT_DIR = "engagement_4class"    # output root

# Create output directories
CLASS_DIRS = {
    "distracted": (0.00, 0.20),
    "disengaged": (0.20, 0.34),
    "moderate": (0.35, 0.67),
    "highly_engaged": (0.67, 1.01),  # 1.01 to include 1.0
}

for cls in CLASS_DIRS:
    os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)

# ===================== READ EXCEL =====================
df = pd.read_excel(EXCEL_FILE)

label_map = {}

for _, row in df.iterrows():
    video_name = str(row[0]).strip()   # subject_X_Vid_Y.avi
    score = float(row[1])

    assigned_class = None
    for cls, (low, high) in CLASS_DIRS.items():
        if low <= score < high:
            assigned_class = cls
            break

    if assigned_class is None:
        print(f"⚠️ Invalid label {score} for {video_name}")
        continue

    label_map[video_name] = assigned_class

print(f"✅ Loaded {len(label_map)} labels from Excel")

# ===================== PROCESS FRAME FOLDERS =====================
for folder in os.listdir(SOURCE_DIR):
    folder_path = os.path.join(SOURCE_DIR, folder)

    if not os.path.isdir(folder_path):
        continue

    # subject_10_Vid_6_2fps → subject_10_Vid_6
    base_name = folder.replace("_2fps", "")

    matched_video = None
    for ext in [".avi", ".AVI", ".mp4", ".MP4", ".webm", ".wmv"]:
        candidate = base_name + ext
        if candidate in label_map:
            matched_video = candidate
            break

    if matched_video is None:
        print(f"⚠️ No label found for {folder}")
        continue

    class_name = label_map[matched_video]
    dest_path = os.path.join(OUTPUT_DIR, class_name, folder)

    shutil.copytree(folder_path, dest_path, dirs_exist_ok=True)
    print(f"✔ {folder} → {class_name}")

print("\n🎯 DONE: 4-class engagement classification completed")