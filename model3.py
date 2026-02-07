import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, detrend, resample

# ---------------- CONFIG ----------------
VIDEO_FOLDER = "videos"       # Folder containing your 70 videos
OUTPUT_FOLDER = "rppg_outputs"
FS = 30                       # original video fps
TARGET_FS = 1                 # desired rPPG sample rate (1 sample/sec)
LOW_HZ = 0.7
HIGH_HZ = 4.0
FILTER_ORDER = 4

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------------- SIGNAL UTILITIES ----------------
def bandpass(signal, fs=FS, low=LOW_HZ, high=HIGH_HZ, order=FILTER_ORDER):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, signal)

def normalize(x):
    return (x - np.mean(x)) / np.std(x)

def downsample_signal(signal, original_fs=FS, target_fs=TARGET_FS):
    n_samples = int(len(signal) * target_fs / original_fs)
    return resample(signal, n_samples)

# ---------------- FACE DETECTION ----------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def extract_face_roi(video_path, max_frames=None):
    """Extract face ROI frames from video using OpenCV Haar cascade"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            roi = frame[y:y+h//3, x:x+w]  # Forehead ROI
            frames.append(roi)

        count += 1
        if max_frames and count >= max_frames:
            break

    cap.release()
    return frames

# ---------------- ALGORITHMS ----------------
def CHROM(frames):
    rgb_means = [np.mean(frame.reshape(-1,3), axis=0) for frame in frames]
    C = np.array(rgb_means).T
    r, g, b = C[0,:], C[1,:], C[2,:]

    r = detrend(normalize(r))
    g = detrend(normalize(g))
    b = detrend(normalize(b))

    X = 3*r - 2*g
    Y = 1.5*r + g - 1.5*b
    alpha = np.std(X)/np.std(Y)
    ppg = X - alpha*Y
    ppg = bandpass(ppg)
    return ppg

def POS(frames):
    rgb_means = [np.mean(frame.reshape(-1,3), axis=0) for frame in frames]
    C = np.array(rgb_means).T  # shape (3,T)

    Cn = (C - np.mean(C, axis=1, keepdims=True)) / np.std(C, axis=1, keepdims=True)
    S1 = Cn[0,:] - Cn[1,:]
    S2 = Cn[0,:] + Cn[1,:] - 2*Cn[2,:]
    alpha = np.std(S1)/np.std(S2)
    ppg = S1 - alpha*S2
    ppg = bandpass(ppg)
    return ppg

def GREEN(frames):
    g_signal = [np.mean(frame[:,:,1]) for frame in frames]
    g_signal = detrend(normalize(np.array(g_signal)))
    g_signal = bandpass(g_signal)
    return g_signal

def save_signal_csv(signal, filename):
    df = pd.DataFrame({"rppg_signal": signal})
    df.to_csv(filename, index=False)

def save_signal_json(signal, filename):
    df = pd.DataFrame({"rppg_signal": signal})
    df.to_json(filename, orient="records")

def plot_signal(signal, title="rPPG Signal"):
    t = np.arange(len(signal))/TARGET_FS
    plt.figure(figsize=(12,4))
    plt.plot(t, signal)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ---------------- MAIN LOOP ----------------
videos = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith((".mp4",".avi",".mov",".webm"))]

for video_file in videos:
    video_path = os.path.join(VIDEO_FOLDER, video_file)
    print(f"Processing {video_file}...")
    frames = extract_face_roi(video_path)

    if len(frames) == 0:
        print(f"No face detected in {video_file}, skipping...")
        continue

    # Run algorithms
    chrom_ppg = CHROM(frames)
    pos_ppg = POS(frames)
    green_ppg = GREEN(frames)

    # Downsample to 1 Hz
    chrom_ppg_1hz = downsample_signal(chrom_ppg)
    pos_ppg_1hz = downsample_signal(pos_ppg)
    green_ppg_1hz = downsample_signal(green_ppg)

    # Save CSV and JSON
    base_name = os.path.splitext(video_file)[0]
    save_signal_csv(chrom_ppg_1hz, os.path.join(OUTPUT_FOLDER, f"{base_name}_CHROM_1Hz.csv"))
    save_signal_csv(pos_ppg_1hz, os.path.join(OUTPUT_FOLDER, f"{base_name}_POS_1Hz.csv"))
    save_signal_csv(green_ppg_1hz, os.path.join(OUTPUT_FOLDER, f"{base_name}_GREEN_1Hz.csv"))

    save_signal_json(chrom_ppg_1hz, os.path.join(OUTPUT_FOLDER, f"{base_name}_CHROM_1Hz.json"))
    save_signal_json(pos_ppg_1hz, os.path.join(OUTPUT_FOLDER, f"{base_name}_POS_1Hz.json"))
    save_signal_json(green_ppg_1hz, os.path.join(OUTPUT_FOLDER, f"{base_name}_GREEN_1Hz.json"))

    # Optional: plot first video only
    if video_file == videos[0]:
        plot_signal(chrom_ppg_1hz, title=f"{video_file} - CHROM 1Hz")
        plot_signal(pos_ppg_1hz, title=f"{video_file} - POS 1Hz")
        plot_signal(green_ppg_1hz, title=f"{video_file} - GREEN 1Hz")

print("All videos processed. rPPG signals saved in:", OUTPUT_FOLDER)
