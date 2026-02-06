import tensorflow as tf
import time

# ================== GPU CHECK ==================
gpus = tf.config.list_physical_devices("GPU")
print(f"🚀 GPUs detected: {len(gpus)}")
for gpu in gpus:
    print("   ", gpu)

if not gpus:
    raise RuntimeError("❌ No GPU detected. Check CUDA / drivers.")

# Enable memory growth
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Use tensor cores
tf.keras.mixed_precision.set_global_policy("mixed_float16")

# ================== STRESS PARAMETERS ==================
MATRIX_SIZE = 16384   # Try 8192 first if unsure
ITERATIONS = 200

print("🔥 Starting GPU stress test...")
start = time.time()

# ================== STRESS LOOP ==================
with tf.device("/GPU:0"):
    a = tf.random.normal((MATRIX_SIZE, MATRIX_SIZE))
    b = tf.random.normal((MATRIX_SIZE, MATRIX_SIZE))

    for i in range(ITERATIONS):
        c = tf.matmul(a, b)
        tf.reduce_sum(c).numpy()   # force execution
        if i % 10 == 0:
            print(f"Iteration {i}/{ITERATIONS}")

end = time.time()

print(f"✅ Stress test finished in {end - start:.2f} seconds")