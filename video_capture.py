# !pip install opencv-python-headless

import cv2
import os

def process_video(video_path, output_dir, frames_after_motion=30):
    # 비디오 파일 열기
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error opening video file")
        return

    # 비디오 속도 계산
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Frames per second: {fps}")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    print(f"Total frames: {frame_count}")
    print(f"Video duration (seconds): {duration}")

    frame_idx = 0
    motion_detected = False
    motion_detected_frame = 0
    motion_stopped_frame = 0

    # 첫 번째 프레임 읽기
    ret, prev_frame = cap.read()
    if not ret:
        print("Error reading the first frame")
        return
    
#     # 첫 번째 프레임 캡쳐
#     capture_path = os.path.join(output_dir, f"bag1_0.jpg")
#     cv2.imwrite(capture_path, prev_frame)
#     print(f"Captured {capture_path} (initial no motion)")
        
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # 프레임 간 차이 계산
        frame_delta = cv2.absdiff(prev_gray, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 움직임 감지
        if any(cv2.contourArea(contour) > 500 for contour in contours):
            motion_detected = True
            motion_detected_frame = frame_idx
        else:
            if motion_detected:
                motion_stopped_frame = frame_idx
                motion_detected = False

        # 움직임이 멈췄을 때 프레임 캡처
        if motion_stopped_frame and (frame_idx == motion_stopped_frame + frames_after_motion):
            fnum = '{0:05d}'.format(frame_idx)
            capture_path = os.path.join(output_dir, f"bag_{fnum}.jpg")
            cv2.imwrite(capture_path, frame)
            print(f"Captured {capture_path}")
            motion_stopped_frame = 0  # 캡처 후 초기화

        prev_gray = gray
        frame_idx += 1

    cap.release()

# 비디오 파일 경로와 출력 디렉토리 설정
video_path = './test_video.mp4'
output_dir = './final/'

# 비디오 처리 함수 호출
process_video(video_path, output_dir)
