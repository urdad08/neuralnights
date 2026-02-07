import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# ================= CONFIG =================
DATA_DIR = "engagement_4class"   # ✅ ROOT FOLDER
IMG_SIZE = (100, 100)
BATCH_SIZE = 32
EPOCHS = 1
SEED = 42

print("🚀 GPUs Available:", tf.config.list_physical_devices("GPU"))

# ================= MIXED PRECISION =================
tf.keras.mixed_precision.set_global_policy("mixed_float16")

# ================= DATASETS =================
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",          # ✅ MULTICLASS
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)

print("📂 Class order used by model:", class_names)

# ================= NORMALIZATION =================
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds   = val_ds.map(lambda x, y: (normalization_layer(x), y))

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds   = val_ds.prefetch(tf.data.AUTOTUNE)

# ================= MODEL =================
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(100, 100, 3)),

    tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),

    # 🔥 4-CLASS OUTPUT (float32 for mixed precision safety)
    tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ================= CALLBACKS =================
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "best_engagement_model.h5",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )
]

# ================= TRAIN =================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ================= EVALUATION =================
loss, acc = model.evaluate(val_ds, verbose=0)
print(f"\n🎯 Validation Accuracy: {acc * 100:.2f}%")

# ================= CONFUSION MATRIX =================
y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    preds = np.argmax(preds, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(preds)

cm = confusion_matrix(y_true, y_pred)

print("\n📊 Confusion Matrix:")
print(cm)

print("\n📋 Classification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=class_names
))

np.savetxt("confusion_matrix.txt", cm, fmt="%d")









# # import os
# # import tensorflow as tf
# # import numpy as np
# # from multiprocessing import cpu_count
# # from sklearn.model_selection import KFold
# # from sklearn.metrics import confusion_matrix, classification_report
 
# # DATA_DIR = "engagement_4class"
# # IMG_SIZE = (100, 100)
# # BATCH_SIZE = 64
# # EPOCHS = 4
# # N_SPLITS = 5
# # SEED = 42

# # CLASS_NAMES = ["distracted", "disengaged", "moderate", "highly_engaged"]
 
# # print(f"⚡ Using {cpu_count()} CPU cores")

# # gpus = tf.config.list_physical_devices("GPU")
# # print(f"🚀 GPUs available: {len(gpus)}")
# # for gpu in gpus:
# #     print(f"   - {gpu.name}")
 
# # strategy = tf.distribute.MirroredStrategy()
# # print(f"🔥 Using strategy: {strategy.__class__.__name__}")
 
# # tf.keras.mixed_precision.set_global_policy("mixed_float16")
 
# # image_paths = []
# # labels = []

# # for idx, cls in enumerate(CLASS_NAMES):
# #     cls_dir = os.path.join(DATA_DIR, cls)
# #     for img in os.listdir(cls_dir):
# #         image_paths.append(os.path.join(cls_dir, img))
# #         labels.append(idx)

# # image_paths = np.array(image_paths)
# # labels = np.array(labels)

# # print(f"🎥 Total videos found: {len(image_paths)}")
 
# # def load_image(path, label):
# #     img = tf.io.read_file(path)
# #     img = tf.image.decode_jpeg(img, channels=3)
# #     img = tf.image.resize(img, IMG_SIZE)
# #     img = img / 255.0
# #     return img, label

# # def make_dataset(paths, labels, training=True):
# #     ds = tf.data.Dataset.from_tensor_slices((paths, labels))
# #     ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
# #     if training:
# #         ds = ds.shuffle(1000, seed=SEED)
# #     ds = ds.batch(BATCH_SIZE)
# #     ds = ds.prefetch(tf.data.AUTOTUNE)
# #     return ds
 
# # def build_model():
# #     model = tf.keras.Sequential([
# #         tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same",
# #                                input_shape=(100, 100, 3)),
# #         tf.keras.layers.MaxPooling2D(),

# #         tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
# #         tf.keras.layers.MaxPooling2D(),

# #         tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
# #         tf.keras.layers.MaxPooling2D(),

# #         tf.keras.layers.Flatten(),
# #         tf.keras.layers.Dense(256, activation="relu"),
# #         tf.keras.layers.Dropout(0.5),

# #         tf.keras.layers.Dense(4, activation="softmax", dtype="float32")
# #     ])

# #     model.compile(
# #         optimizer=tf.keras.optimizers.Adam(1e-4),
# #         loss="sparse_categorical_crossentropy",
# #         metrics=["accuracy"]
# #     )
# #     return model
 
# # kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

# # all_y_true = []
# # all_y_pred = []

# # fold = 1
# # for train_idx, val_idx in kf.split(image_paths):
# #     print(f"\n==================== FOLD {fold} ====================")

# #     train_ds = make_dataset(image_paths[train_idx], labels[train_idx], True)
# #     val_ds = make_dataset(image_paths[val_idx], labels[val_idx], False)

# #     with strategy.scope():
# #         model = build_model()

# #     model.summary()

# #     model.fit(
# #         train_ds,
# #         validation_data=val_ds,
# #         epochs=EPOCHS,
# #         verbose=1
# #     )
 
# #     y_true = []
# #     y_pred = []

# #     for imgs, lbls in val_ds:
# #         preds = model.predict(imgs, verbose=0)
# #         preds = np.argmax(preds, axis=1)

# #         y_true.extend(lbls.numpy())
# #         y_pred.extend(preds)

# #     all_y_true.extend(y_true)
# #     all_y_pred.extend(y_pred)

# #     fold += 1
 
# # cm = confusion_matrix(all_y_true, all_y_pred)

# # print("\n📊 Confusion Matrix:")
# # print(cm)

# # print("\n📋 Classification Report:")
# # print(classification_report(
# #     all_y_true,
# #     all_y_pred,
# #     target_names=CLASS_NAMES
# # ))

# # acc = np.mean(np.array(all_y_true) == np.array(all_y_pred))
# # print(f"\n🎯 Final Cross-Validated Accuracy: {acc * 100:.2f}%")
# import os
# import tensorflow as tf
# import numpy as np
# from multiprocessing import cpu_count
# from sklearn.model_selection import KFold
# from sklearn.metrics import confusion_matrix, classification_report

# # ===================== CONFIG =====================
# DATA_DIR = "engagement_4class"
# IMG_SIZE = (100, 100)
# BATCH_SIZE = 64
# EPOCHS = 4
# N_SPLITS = 5
# SEED = 42

# CLASS_NAMES = ["distracted", "disengaged", "moderate", "highly_engaged"]

# # ===================== SYSTEM INFO =====================
# print(f"⚡ Using {cpu_count()} CPU cores")

# gpus = tf.config.list_physical_devices("GPU")
# print(f"🚀 GPUs available: {len(gpus)}")
# for gpu in gpus:
#     print(f"   - {gpu.name}")

# strategy = tf.distribute.MirroredStrategy()
# print(f"🔥 Using strategy: {strategy.__class__.__name__}")

# tf.keras.mixed_precision.set_global_policy("mixed_float16")

# # ===================== COLLECT IMAGE PATHS (FIXED) =====================
# image_paths = []
# labels = []

# for label_idx, cls in enumerate(CLASS_NAMES):
#     class_dir = os.path.join(DATA_DIR, cls)
#     if not os.path.isdir(class_dir):
#         continue

#     for video_folder in os.listdir(class_dir):
#         video_path = os.path.join(class_dir, video_folder)
#         if not os.path.isdir(video_path):
#             continue

#         for img in os.listdir(video_path):
#             if img.lower().endswith((".jpg", ".jpeg", ".png")):
#                 image_paths.append(os.path.join(video_path, img))
#                 labels.append(label_idx)

# image_paths = np.array(image_paths)
# labels = np.array(labels)

# print(f"🎥 Total frames found: {len(image_paths)}")

# # ===================== DATASET PIPELINE =====================
# def load_image(path, label):
#     try:
#         img = tf.io.read_file(path)

#         # Skip empty files
#         if tf.size(img) == 0:
#             raise ValueError("Empty file")

#         img = tf.image.decode_jpeg(img, channels=3)
#         img = tf.image.resize(img, IMG_SIZE)
#         img = tf.cast(img, tf.float32) / 255.0
#         return img, label

#     except Exception:
#         # Return a dummy image & label (will be filtered later)
#         return tf.zeros([IMG_SIZE[0], IMG_SIZE[1], 3]), -1

# def make_dataset(paths, labels, training=True):
#     ds = tf.data.Dataset.from_tensor_slices((paths, labels))
#     ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

#     # 🔥 FILTER OUT BAD SAMPLES
#     ds = ds.filter(lambda x, y: tf.not_equal(y, -1))

#     if training:
#         ds = ds.shuffle(1000, seed=SEED)

#     ds = ds.batch(BATCH_SIZE)
#     ds = ds.prefetch(tf.data.AUTOTUNE)
#     return ds

# # ===================== MODEL =====================
# def build_model():
#     model = tf.keras.Sequential([
#         tf.keras.layers.Input(shape=(100, 100, 3)),

#         tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
#         tf.keras.layers.MaxPooling2D(),

#         tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
#         tf.keras.layers.MaxPooling2D(),

#         tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
#         tf.keras.layers.MaxPooling2D(),

#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(256, activation="relu"),
#         tf.keras.layers.Dropout(0.5),

#         tf.keras.layers.Dense(4, activation="softmax", dtype="float32")
#     ])

#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(1e-4),
#         loss="sparse_categorical_crossentropy",
#         metrics=["accuracy"]
#     )
#     return model

# # ===================== K-FOLD TRAINING =====================
# kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

# all_y_true = []
# all_y_pred = []

# fold = 1
# for train_idx, val_idx in kf.split(image_paths):
#     print(f"\n==================== FOLD {fold} ====================")

#     train_ds = make_dataset(image_paths[train_idx], labels[train_idx], True)
#     val_ds   = make_dataset(image_paths[val_idx], labels[val_idx], False)

#     with strategy.scope():
#         model = build_model()

#     model.summary()

#     model.fit(
#         train_ds,
#         validation_data=val_ds,
#         epochs=EPOCHS,
#         verbose=1
#     )

#     # ---- Evaluation ----
#     y_true = []
#     y_pred = []

#     for imgs, lbls in val_ds:
#         preds = model.predict(imgs, verbose=0)
#         preds = np.argmax(preds, axis=1)

#         y_true.extend(lbls.numpy())
#         y_pred.extend(preds)

#     all_y_true.extend(y_true)
#     all_y_pred.extend(y_pred)

#     fold += 1

# # ===================== FINAL METRICS =====================
# cm = confusion_matrix(all_y_true, all_y_pred)

# print("\n📊 Confusion Matrix:")
# print(cm)

# print("\n📋 Classification Report:")
# print(classification_report(
#     all_y_true,
#     all_y_pred,
#     target_names=CLASS_NAMES
# ))

# accuracy = np.mean(np.array(all_y_true) == np.array(all_y_pred))
# print(f"\n🎯 Final Cross-Validated Accuracy: {accuracy * 100:.2f}%")
# import os
# from PIL import Image
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, models
# from sklearn.model_selection import train_test_split
# from tqdm import tqdm

# # =========================
# # CONFIG
# # =========================
# DATASET_DIR = "engagement_4class"
# BATCH_SIZE = 16
# EPOCHS = 10
# LR = 1e-4
# IMG_SIZE = 224
# NUM_CLASSES = 4

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CLASS_NAMES = ["disengaged", "moderate", "highly_engaged", "distracted"]
# CLASS_TO_IDX = {c: i for i, c in enumerate(CL_NAMES)}

# # =========================
# # SAFE DATASET
# # =========================
# class EngagementDataset(Dataset):
#     def __init__(self, samples, transform=None):
#         self.samples = samples
#         self.transform = transform

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         img_path, label = self.samples[idx]
#         try:
#             img = Image.open(img_path).convert("RGB")
#         except Exception:
#             # fallback black image if corrupted
#             img = Image.new("RGB", (IMG_SIZE, IMG_SIZE))

#         if self.transform:
#             img = self.transform(img)

#         return img, label


# def collect_samples(root_dir):
#     samples = []
#     for class_name in CLASS_NAMES:
#         class_dir = os.path.join(root_dir, class_name)
#         for root, _, files in os.walk(class_dir):
#             for f in files:
#                 if f.lower().endswith((".jpg", ".png", ".jpeg")):
#                     samples.append(
#                         (os.path.join(root, f), CLASS_TO_IDX[class_name])
#                     )
#     return samples


# # =========================
# # TRANSFORMS
# # =========================
# train_tfms = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.RandomHorizontalFlip(),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     )
# ])

# val_tfms = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     )
# ])

# # =========================
# # LOAD DATA
# # =========================
# all_samples = collect_samples(DATASET_DIR)
# train_samples, val_samples = train_test_split(
#     all_samples, test_size=0.2, random_state=42, stratify=[s[1] for s in all_samples]
# )

# train_ds = EngagementDataset(train_samples, train_tfms)
# val_ds = EngagementDataset(val_samples, val_tfms)

# train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
# val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# # =========================
# # MODEL (SAFE TRANSFER LEARNING)
# # =========================
# model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
# model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
# model = model.to(DEVICE)

# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# # =========================
# # TRAINING LOOP
# # =========================
# for epoch in range(EPOCHS):
#     model.train()
#     train_loss = 0.0

#     for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
#         imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

#         optimizer.zero_grad()
#         outputs = model(imgs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         train_loss += loss.item()

#     model.eval()
#     correct = total = 0

#     with torch.no_grad():
#         for imgs, labels in val_loader:
#             imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
#             outputs = model(imgs)
#             preds = outputs.argmax(dim=1)
#             correct += (preds == labels).sum().item()
#             total += labels.size(0)

#     acc = 100 * correct / total
#     print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Val Acc={acc:.2f}%")

# # =========================
# # SAVE MODEL
# # =========================
# torch.save(model.state_dict(), "engagement_model.pth")
# print("Model saved as engagement_model.pth")