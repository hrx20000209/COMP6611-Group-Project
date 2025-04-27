import cv2
import time

def video_data_collector(video_queue):
    cap = cv2.VideoCapture(0)  # 默认摄像头

    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    print("[INFO] Starting video capture...")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            timestamp = time.time()
            # 可选：你可以调整 frame 大小以减少内存负担
            # frame = cv2.resize(frame, (320, 240))

            video_queue.put({
                "timestamp": timestamp,
                "frame": frame
            })

            # 控制帧率：约 30 FPS
            # time.sleep(0.3)

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        print("[INFO] Video process terminated.")
