# yaml 파일 생성
data_yaml = {
    'train': '../bag/train/images',
    'val': '../bag/val/images',
    'nc': 1,  
    'names': ['bag'] 
}

with open('../bag/bag_custom.yaml', 'w') as outfile:
    yaml.dump(data_yaml, outfile, default_flow_style=False)

# !python train.py --img 640 --batch 16 --epochs 100 --data ../bag/bag_custom.yaml --weights yolov5s.pt

import cv2
import torch
import os
from glob import glob

# 학습된 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'custom', path='path/to/best.pt', source='local')

def detect_and_crop(image_path, output_folder):
    # 이미지 로드
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image {image_path}")
        return

    # 객체 검출
    results = model(img)

    # labels = results.xyxyn[0][:, -1]
    # boxes = results.xyxyn[0][:, :-1]
    # names = results.names

    for i, label in enumerate(labels):
        if names[int(label)] == 'bag':
            box = boxes[i]
            x1, y1, x2, y2 = box[0].item(), box[1].item(), box[2].item(), box[3].item()
            h, w, _ = img.shape

            # 원래 이미지 크기로 변환
            x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)

            # 이미지 크롭 및 저장
            cropped_img = img[y1:y2, x1:x2]
            base_filename = os.path.basename(image_path)
            filename, ext = os.path.splitext(base_filename)
            output_path = os.path.join(output_folder, f"{filename}_bag_{i}{ext}")
            cv2.imwrite(output_path, cropped_img)
            print(f"Saved cropped image to {output_path}")

# 이미지가 저장된 폴더 경로 및 출력 폴더 설정
input_folder = 'path/to/your/image/folder'
output_folder = 'path/to/save/cropped/images'

# 출력 폴더가 존재하지 않으면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 폴더 내의 모든 이미지 파일 경로 가져오기
image_paths = glob(os.path.join(input_folder, '*.jpg')) + glob(os.path.join(input_folder, '*.png')) + glob(os.path.join(input_folder, '*.jpeg'))

# 각 이미지에 대해 가방 검출 및 크롭
for image_path in image_paths:
    detect_and_crop(image_path, output_folder)
