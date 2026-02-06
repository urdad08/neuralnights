import os
import shutil
import pandas as pd
 
EXCEL_FILE = "dataset/train_labels.xlsx"      
SOURCE_DIR = "frames_100x100"
DEST_0 = "label0"
DEST_1 = "label1"

os.makedirs(DEST_0, exist_ok=True)
os.makedirs(DEST_1, exist_ok=True)
 
df = pd.read_excel(EXCEL_FILE)    

labels = {}

for _, row in df.iterrows():
    filename = str(row.iloc[0]).strip()
    label = float(row.iloc[1])

    binary_label = 0 if label <= 0.33 else 1
    labels[filename] = binary_label

print(f"✅ Loaded {len(labels)} labels from Excel")
 
for folder in os.listdir(SOURCE_DIR):
    folder_path = os.path.join(SOURCE_DIR, folder)

    if not os.path.isdir(folder_path):
        continue

    # subject_1_Vid_5_2fps → subject_1_Vid_5
    base_name = folder.replace("_2fps", "")

    matched_file = None
    for ext in [".avi", ".AVI", ".mp4", ".MP4", ".webm", ".wmv"]:
        candidate = base_name + ext
        if candidate in labels:
            matched_file = candidate
            break

    if matched_file is None:
        print(f"⚠️ No label found for {folder}")
        continue

    target_dir = DEST_0 if labels[matched_file] == 0 else DEST_1
    dest_path = os.path.join(target_dir, folder)

    shutil.copytree(folder_path, dest_path, dirs_exist_ok=True)

    print(f"✔ {folder} → {'label0' if labels[matched_file] == 0 else 'label1'}")

print("🎯 DONE: All folders classified successfully")