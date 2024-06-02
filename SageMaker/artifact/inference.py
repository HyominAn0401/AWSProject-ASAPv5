import os
import json
import cv2
import torch
from PIL import Image
import boto3
import io
import numpy as np

s3_client = boto3.client('s3')

def model_fn(model_dir):
    model_path = os.path.join(model_dir, 'yolov5_custom.pt')
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    model.eval()
    return model

def input_fn(request_body, content_type='application/json'):
    if content_type == 'application/json':
        request = json.loads(request_body)
        bucket = request['bucket']
        video_path = request['video_path']
        local_video_path = '/tmp/video.mp4'
        s3_client.download_file(bucket, video_path, local_video_path)
        output_dir = '/tmp/cap_img/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        process_video(local_video_path, output_dir)
        return output_dir
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_object, model):
    output_dir = input_object
    results = []
    for frame_file in os.listdir(output_dir):
        frame_path = os.path.join(output_dir, frame_file)
        img = Image.open(frame_path)
        results.append(model(img))
    return results

def output_fn(prediction, content_type='application/json'):
    if content_type == 'application/json':
        results = []
        for result in prediction:
            results.append(result.pandas().xyxy[0].to_json(orient="records"))
        return json.dumps(results)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def process_video(video_path, output_dir, frames_after_motion=30):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file")

    frame_idx = 0
    motion_detected = False
    motion_detected_frame = 0
    motion_stopped_frame = 0

    ret, prev_frame = cap.read()
    if not ret:
        raise ValueError("Error reading the first frame")
    
    capture_path = os.path.join(output_dir, f"frame_00000.jpg")
    cv2.imwrite(capture_path, prev_frame)
    upload_to_s3(capture_path, 'asapv5-s3', 'processed_images/frame_00000.jpg')
        
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        frame_delta = cv2.absdiff(prev_gray, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if any(cv2.contourArea(contour) > 500 for contour in contours):
            motion_detected = True
            motion_detected_frame = frame_idx
        else:
            if motion_detected:
                motion_stopped_frame = frame_idx
                motion_detected = False

        if motion_stopped_frame and (frame_idx == motion_stopped_frame + frames_after_motion):
            fnum = '{0:05d}'.format(frame_idx)
            capture_path = os.path.join(output_dir, f"frame_{fnum}.jpg")
            cv2.imwrite(capture_path, frame)
            upload_to_s3(capture_path, 'asapv5-s3', f'processed_images/frame_{fnum}.jpg')
            motion_stopped_frame = 0

        prev_gray = gray
        frame_idx += 1

    cap.release()

def upload_to_s3(file_path, bucket, key):
    s3_client.upload_file(file_path, bucket, key)
