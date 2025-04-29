import numpy as np
import cv2
import argparse

def visualize_video(video_data_list):
    for frame_data in video_data_list:
        frame = frame_data["frame"]
        timestamp = frame_data["timestamp"]

        # 显示时间戳
        annotated = frame.copy()
        cv2.putText(annotated, f"Timestamp: {timestamp:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Video Frame", annotated)
        key = cv2.waitKey(33)  # 大约 30 FPS
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to the .npz file")
    args = parser.parse_args()

    data = np.load(args.file, allow_pickle=True)
    video_data = data['video']

    print(f"[INFO] Loaded {len(video_data)} video frames.")
    visualize_video(video_data)
