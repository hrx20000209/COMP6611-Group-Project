import multiprocessing as mp
import time
import os
import numpy as np

from datetime import datetime

# 导入两个子进程模块（你后面会创建这两个文件）
from radar_process import radar_data_collector
from video_process import video_data_collector

SAMPLE_INTERVAL = 1.5  # seconds
SAVE_DIR = "samples"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def save_sample(sample_index, radar_data, video_data):
    filename = os.path.join(SAVE_DIR, f"./squating/sample_{sample_index:05d}.npz")
    np.savez_compressed(filename, radar=radar_data, video=video_data)
    print(f"[INFO] Saved {filename}")

if __name__ == '__main__':
    mp.set_start_method('spawn')  # Better for compatibility

    radar_queue = mp.Queue()
    video_queue = mp.Queue()

    radar_proc = mp.Process(target=radar_data_collector, args=(radar_queue,))
    video_proc = mp.Process(target=video_data_collector, args=(video_queue,))

    radar_proc.start()
    video_proc.start()

    try:
        sample_index = 88
        while sample_index < 120:
            t_start = time.time()
            radar_data_buffer = []
            video_data_buffer = []

            # 收集2秒钟的数据
            while time.time() - t_start < SAMPLE_INTERVAL:
                try:
                    radar_data = radar_queue.get(timeout=0.1)
                    radar_data_buffer.append(radar_data)
                except:
                    pass
                try:
                    video_frame = video_queue.get(timeout=0.1)
                    video_data_buffer.append(video_frame)
                except:
                    pass

            # 保存 sample（可扩展为带时间戳的数据结构）
            save_sample(sample_index, radar_data_buffer, video_data_buffer)
            sample_index += 1

    except KeyboardInterrupt:
        print("[INFO] Stopping processes...")
        radar_proc.terminate()
        video_proc.terminate()
        radar_proc.join()
        video_proc.join()
        print("[INFO] Exited cleanly.")
