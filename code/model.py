import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
 
DATA_DIR = "dataset/labels"
IMG_SIZE = (100, 100)
BATCH_SIZE = 32
EPOCHS = 2
SEED = 42

print("GPUs Available:", tf.config.list_physical_devices('GPU'))

 
tf.keras.mixed_precision.set_global_policy("mixed_float16")
 
 
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

class_names = train_ds.class_names
print("Classes:", class_names)

# Normalize images
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds   = val_ds.map(lambda x, y: (normalization_layer(x), y))

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds   = val_ds.prefetch(tf.data.AUTOTUNE)
 
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same",
                           input_shape=(100, 100, 3)),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()
 
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "best_cnn_model.h5",
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

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)
 
loss, acc = model.evaluate(val_ds)
print(f"\n🎯 Test Accuracy: {acc * 100:.2f}%")
 
y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    preds = (preds > 0.5).astype(int).flatten()

    y_true.extend(labels.numpy().astype(int))
    y_pred.extend(preds)
 
cm = confusion_matrix(y_true, y_pred)

print("\n📊 Confusion Matrix:")
print(cm)

print("\n📋 Classification Report:")
print(classification_report(
    y_true, y_pred,
    target_names=class_names
))
np.savetxt("confusion_matrix.txt", cm, fmt="%d")