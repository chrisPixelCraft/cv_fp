import cv2
import numpy as np
import tqdm

def total_frames(video_path):
    # 獲取視頻的總幀數
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames

def adjust_exposure(frame, value):
    # 調整曝光
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    # v = np.clip(v, 0, 256)
    final_hsv = cv2.merge((h, s, v))
    frame = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return frame

def fisheye_effect(frame):
    # 魚眼效果
    height, width = frame.shape[:2]
    K = np.array([[width, 0, width / 2],
                  [0, width, height / 2],
                  [0, 0, 1]])
    D = np.array([15, 15, 15, 5.0])  # 魚眼參數
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (width, height), cv2.CV_16SC2)
    frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return frame

def process_video(input_file, output_file):
    cap = cv2.VideoCapture(input_file)
    # cap = cv.resize(cap, (224, 224))
    ct = 0
    while(1):
        print(ct)
        ct = ct + 1
        ret, frame2 = cap.read()
        if not ret:
            print('No frames grabbed!')
            break

        frame2 = adjust_exposure(frame2, 0)
        frame2 = fisheye_effect(frame2)

        cv2.imshow('frame2', frame2)
        
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('opticalfb.png', frame2)
            cv2.imwrite('opticalhsv.png', s)
            prvs = next
    cv2.destroyAllWindows()


# 調用函數
if __name__ == '__main__':
    print('hi')
    process_video('../data/raw_videos/01.mp4', '../data/aug_videos/output.mp4')
