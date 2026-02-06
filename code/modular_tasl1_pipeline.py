# =========================================================
# TASK 1 : END-TO-END PIPELINE (SINGLE FILE, MODULAR)
# =========================================================

import os
import cv2
import dlib
import time
import numpy as np
import tensorflow as tf
from multiprocessing import Pool, cpu_count
from sklearn.metrics import confusion_matrix, classification_report

# =========================================================
# CONFIG
# =========================================================
RAW_VIDEO_DIR = "dataset/Train"
VIDEO_2FPS_DIR = "dataset_2fps"
FACE_DIR = "output_faces"
FRAME_100_DIR = "frames_100x100"
LABEL_DIR = "dataset/labels"   # for training
IMG_SIZE = (100, 100)
TARGET_FPS = 2
EPOCHS = 2
BATCH_SIZE = 32
SEED = 42

VIDEO_EXTS = (".avi", ".mp4", ".wmv", ".mov", ".webm")

# =========================================================
# STAGE 1: VIDEO → 2 FPS
# =========================================================
def convert_to_2fps():
    os.makedirs(VIDEO_2FPS_DIR, exist_ok=True)
    frame_gap_ms = 1000 / TARGET_FPS

    for video in os.listdir(RAW_VIDEO_DIR):
        if not video.lower().endswith(VIDEO_EXTS):
            continue

        cap = cv2.VideoCapture(os.path.join(RAW_VIDEO_DIR, video))
        if not cap.isOpened():
            continue

        w, h = int(cap.get(3)), int(cap.get(4))
        out_name = os.path.splitext(video)[0] + "_2fps.mp4"
        out = cv2.VideoWriter(
            os.path.join(VIDEO_2FPS_DIR, out_name),
            cv2.VideoWriter_fourcc(*"mp4v"),
            TARGET_FPS, (w, h)
        )

        next_ts = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            ts = cap.get(cv2.CAP_PROP_POS_MSEC)
            if ts >= next_ts:
                out.write(frame)
                next_ts += frame_gap_ms

        cap.release()
        out.release()
        print(f"✅ 2FPS: {video}")

# =========================================================
# STAGE 2: FACE DETECTION
# =========================================================
detector = dlib.get_frontal_face_detector()

def process_video_faces(video):
    cap = cv2.VideoCapture(os.path.join(VIDEO_2FPS_DIR, video))
    vid = video.rsplit(".", 1)[0]
    out_dir = os.path.join(FACE_DIR, vid)
    os.makedirs(out_dir, exist_ok=True)

    fno = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)

        for i, face in enumerate(faces):
            x1, y1 = max(0, face.left()), max(0, face.top())
            x2, y2 = face.right(), face.bottom()
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            cv2.imwrite(f"{out_dir}/f{fno:05d}_{i}.jpg", crop)

        fno += 1

    cap.release()
    print(f"🙂 Faces extracted: {video}")

def run_face_detection():
    os.makedirs(FACE_DIR, exist_ok=True)
    videos = [v for v in os.listdir(VIDEO_2FPS_DIR) if v.lower().endswith(VIDEO_EXTS)]
    print(f"🎥 Total videos found: {len(videos)}")
    print(f"⚡ Using {cpu_count()} CPU cores")
    with Pool(cpu_count()) as p:
        p.map(process_video_faces, videos)

# =========================================================
# STAGE 3: RESIZE → 100×100
# =========================================================
def uniform_resize():
    os.makedirs(FRAME_100_DIR, exist_ok=True)

    for vid in os.listdir(FACE_DIR):
        in_dir = os.path.join(FACE_DIR, vid)
        out_dir = os.path.join(FRAME_100_DIR, vid)
        os.makedirs(out_dir, exist_ok=True)

        for img in os.listdir(in_dir):
            img_path = os.path.join(in_dir, img)
            im = cv2.imread(img_path)
            if im is None:
                continue

            h, w = im.shape[:2]
            scale = 100 / min(h, w)
            resized = cv2.resize(im, (int(w*scale), int(h*scale)))
            x = (resized.shape[1] - 100) // 2
            y = (resized.shape[0] - 100) // 2
            crop = resized[y:y+100, x:x+100]

            cv2.imwrite(os.path.join(out_dir, img), crop)

        print(f"🖼️ Resized: {vid}")

# =========================================================
# STAGE 4: CNN MODEL (TASK-1)
# =========================================================
def train_model():
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    print("🚀 GPUs:", tf.config.list_physical_devices("GPU"))

    train_ds = tf.keras.utils.image_dataset_from_directory(
        LABEL_DIR,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        LABEL_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary"
    )

    norm = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x,y:(norm(x),y)).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x,y:(norm(x),y)).prefetch(tf.data.AUTOTUNE)

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32,3,activation="relu",padding="same",input_shape=(100,100,3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64,3,activation="relu",padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128,3,activation="relu",padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1,activation="sigmoid",dtype="float32")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    y_true, y_pred = [], []
    for x,y in val_ds:
        p = (model.predict(x) > 0.5).astype(int)
        y_true.extend(y.numpy())
        y_pred.extend(p)

    print("\n📊 Confusion Matrix")
    print(confusion_matrix(y_true, y_pred))

    print("\n📋 Classification Report")
    print(classification_report(y_true, y_pred))

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    t0 = time.time()
    convert_to_2fps()
    run_face_detection()
    uniform_resize()
    train_model()
    print(f"\n⏱️ Total pipeline time: {(time.time()-t0)/60:.2f} minutes")