# import json
# import boto3

# s3_client = boto3.client('s3')

# bucket_name = 'asapv5-s3'

# def lambda_handler(event, context):
#     try:
#         body = json.loads(event['body'])
#         file_name = body['file_name']
        
#         # Presigned URL
#         presigned_url = s3_client.generate_presigned_url(
#             'put_object',
#             Params={'Bucket': bucket_name, 'Key': f"uploaded_videos/{file_name}"},
#             ExpiresIn=3600
#         )
        
#         return {
#             'statusCode': 200,
#             'body': json.dumps({'presigned_url': presigned_url, 's3_key': f"uploaded_videos/{file_name}"})
#         }
    
#     except Exception as e:
#         return {
#             'statusCode': 500,
#             'body': json.dumps({'message': f"Error generating presigned URL: {str(e)}"})
#         }
import json
import boto3
import cv2
import os
import tempfile

s3_client = boto3.client('s3')
bucket_name = 'asapv5-s3'

def process_video(video_path, output_dir, frames_after_motion=30):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error opening video file")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    frame_idx = 0
    motion_detected = False
    motion_detected_frame = 0
    motion_stopped_frame = 0

    ret, prev_frame = cap.read()
    if not ret:
        print("Error reading the first frame")
        return
    
    capture_path = os.path.join(output_dir, f"frame_0.jpg")
    cv2.imwrite(capture_path, prev_frame)
    upload_to_s3(capture_path, f"processed_images/frame_0.jpg")
        
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
            upload_to_s3(capture_path, f"processed_images/frame_{fnum}.jpg")
            motion_stopped_frame = 0

        prev_gray = gray
        frame_idx += 1

    cap.release()

def upload_to_s3(file_path, s3_key):
    s3_client.upload_file(file_path, bucket_name, s3_key)

def lambda_handler(event, context):
    try:
        body = json.loads(event['body'])
        file_name = body['file_name']
        video_s3_key = f"uploaded_videos/{file_name}"
        
        with tempfile.TemporaryDirectory() as tmpdirname:
            local_video_path = os.path.join(tmpdirname, file_name)
            s3_client.download_file(bucket_name, video_s3_key, local_video_path)
            process_video(local_video_path, tmpdirname)
        
        return {
            'statusCode': 200,
            'body': json.dumps({'message': 'Video processed successfully'})
        }
    
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'message': f"Error processing video: {str(e)}"})
        }
