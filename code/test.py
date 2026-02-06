# import cv2
# import dlib
# import os
# import numpy as np
# from multiprocessing import Pool, cpu_count

# DATASET_DIR = "dataset_2fps"
# OUTPUT_DIR = "output_numpy"

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # Load detector globally (important for multiprocessing)
# detector = dlib.get_frontal_face_detector()

# def process_video(video_name):
#     video_path = os.path.join(DATASET_DIR, video_name)
#     cap = cv2.VideoCapture(video_path)

#     if not cap.isOpened():
#         print(f"❌ Cannot open {video_name}")
#         return

#     data = []
#     frame_no = 0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = detector(gray, 1)

#         for face_id, face in enumerate(faces):
#             x1, y1 = face.left(), face.top()
#             x2, y2 = face.right(), face.bottom()

#             data.append([frame_no, face_id, x1, y1, x2, y2])

#         frame_no += 1

#     cap.release()

#     # Convert to NumPy array
#     data_np = np.array(data, dtype=np.int32)

#     out_name = video_name.rsplit(".", 1)[0] + ".npy"
#     out_path = os.path.join(OUTPUT_DIR, out_name)

#     np.save(out_path, data_np)
#     print(f"✅ Saved: {out_name} | Rows: {data_np.shape[0]}")

# def main():
#     videos = [
#         v for v in os.listdir(DATASET_DIR)
#         if v.lower().endswith((".mp4", ".avi", ".mov",".webm",".wmv"))
#     ]

#     print(f"🎥 Total videos found: {len(videos)}")
#     print(f"⚡ Using {cpu_count()} CPU cores")

#     with Pool(processes=cpu_count()) as pool:
#         pool.map(process_video, videos)

# if __name__ == "__main__":
#     main()

import cv2
import dlib
import os
from multiprocessing import Pool, cpu_count

DATASET_DIR = "dataset_2fps"
OUTPUT_DIR = "output_images_complete"

os.makedirs(OUTPUT_DIR, exist_ok=True)
 
detector = dlib.get_frontal_face_detector()

def process_video(video_name):
    video_path = os.path.join(DATASET_DIR, video_name)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Cannot open {video_name}")
        return

    video_id = video_name.rsplit(".", 1)[0]
    video_out_dir = os.path.join(OUTPUT_DIR, video_id)
    os.makedirs(video_out_dir, exist_ok=True)

    frame_no = 0
    saved_faces = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)

        for face_id, face in enumerate(faces):
            x1, y1 = max(0, face.left()), max(0, face.top())
            x2, y2 = min(frame.shape[1], face.right()), min(frame.shape[0], face.bottom())

            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue

            out_name = f"frame_{frame_no:06d}_face_{face_id}.jpg"
            out_path = os.path.join(video_out_dir, out_name)

            cv2.imwrite(out_path, face_img)
            saved_faces += 1

        frame_no += 1

    cap.release()
    print(f"✅ {video_name}: saved {saved_faces} face images")

def main():
    videos = [
        v for v in os.listdir(DATASET_DIR)
        if v.lower().endswith((".mp4", ".avi", ".mov", ".webm", ".wmv"))
    ]

    print(f"🎥 Total videos found: {len(videos)}")
    print(f"⚡ Using {cpu_count()} CPU cores")

    with Pool(processes=cpu_count()) as pool:
        pool.map(process_video, videos)

if __name__ == "__main__":
    main()