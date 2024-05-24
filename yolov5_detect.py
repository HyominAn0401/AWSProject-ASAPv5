# ! pip3 install boto3
# ! pip install opencv-python

### 데이터 불러오기 및 확인
import boto3

bucket_name = 'asapv5-s3'
prefix = 'aws_dataset/train/images/'

s3 = boto3.client('s3')
train_list = []
continuation_token = None

while True:
    if continuation_token:
        response = s3.list_objects_v2(
            Bucket=bucket_name,
            Prefix=prefix,
            ContinuationToken=continuation_token
        )
    else:
        response = s3.list_objects_v2(
            Bucket=bucket_name,
            Prefix=prefix
        )
    
    if 'Contents' in response:
        for content in response['Contents']:
            train_list.append(content['Key'])
    
    if response.get('IsTruncated'):  # 더 많은 객체가 있을 경우
        continuation_token = response['NextContinuationToken']
    else:
        break

print(f"Total images: {len(train_list)}")

import io
file = io.BytesIO()
file_name = train_list[1]
bucket_obj = bucket.Object(train_list[0])
bucket_obj.download_fileobj(file)

import cv2
import numpy as np
import matplotlib.pyplot as plt

file.seek(0)
img_file = np.fromstring(file.read(),np.uint8)
img = cv2.imdecode(img_file,cv2.IMREAD_COLOR)
plt.imshow(img)

from PIL import Image

file.seek(0)
pil_image = Image.open(file)
pil_image.show()


### YOLO v5

# !git clone https://github.com/ultralytics/yolov5
# !pip install -r yolov5/requirements.txt

# sagemaker에서 numpy 충돌 문제 발생.. 고쳐지긴 함
# !pip uninstall numpy --yes
# !pip install --upgrade pip
# !pip install --upgrade pip setuptools
# !pip install numpy

import os
import boto3
import cv2
import torch
import yaml
from pathlib import Path
from sagemaker.pytorch import PyTorch

# S3 설정
bucket_name = 'asapv5-s3'
prefix_train_images = 'aws_dataset/train/images/'
prefix_train_labels = 'aws_dataset/train/labels/'
prefix_val_images = 'aws_dataset/val/images/'
prefix_val_labels = 'aws_dataset/val/labels/'

s3 = boto3.client('s3')

# 객체 리스트를 가져오는 함수
def list_s3_objects(bucket, prefix):
    object_list = []
    continuation_token = None
    while True:
        if continuation_token:
            response = s3.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix,
                ContinuationToken=continuation_token
            )
        else:
            response = s3.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix
            )
        
        if 'Contents' in response:
            for content in response['Contents']:
                object_list.append(content['Key'])
        
        if response.get('IsTruncated'):  # 더 많은 객체가 있을 경우
            continuation_token = response['NextContinuationToken']
        else:
            break
    return object_list

# 이미지 및 라벨 파일 목록 가져오기
train_image_list = list_s3_objects(bucket_name, prefix_train_images)
train_label_list = list_s3_objects(bucket_name, prefix_train_labels)
val_image_list = list_s3_objects(bucket_name, prefix_val_images)
val_label_list = list_s3_objects(bucket_name, prefix_val_labels)

print(f"Total train images: {len(train_image_list)}")
print(f"Total train labels: {len(train_label_list)}")
print(f"Total val images: {len(val_image_list)}")
print(f"Total val labels: {len(val_label_list)}")

# 학습 및 검증 데이터 준비
os.makedirs('data/train/images', exist_ok=True)
os.makedirs('data/train/labels', exist_ok=True)
os.makedirs('data/val/images', exist_ok=True)
os.makedirs('data/val/labels', exist_ok=True)

print('Data download')
# S3에서 로컬로 데이터 다운로드
def download_s3_files(bucket, file_list, local_dir):
    for file in file_list:
        local_path = os.path.join(local_dir, os.path.basename(file))
        s3.download_file(bucket, file, local_path)

download_s3_files(bucket_name, train_image_list, 'data/train/images')
download_s3_files(bucket_name, train_label_list, 'data/train/labels')
download_s3_files(bucket_name, val_image_list, 'data/val/images')
download_s3_files(bucket_name, val_label_list, 'data/val/labels')

print('Create yaml')
# yaml 파일 생성
data_yaml = {
    'train': '../data/train/images',
    'val': '../data/val/images',
    'nc': 42,  
    'names': ['Hammer', 'SSD','Alcohol','Spanner','Axe','Awl','Throwing Knife','Firecracker',
              'Thinner','Plier', 'Match', 'Smart Phone', 'Scissors', 'Tablet PC', 'Solid Fuel', 'Bat', 
              'Portable Gas','Nail Clippers','Knife','Metal Pipe','Electronic Cigarettes(Liquid)',
              'Supplymentary Battery','Bullet','Gun Parts','USB','Liquid','Aerosol','Screwdriver',
              'Chisel','Handcuffs','Lighter','HDD','Electronic Cigarettes','Battery','Gun','Laptop',
              'Saw','Zippo Oil','Stun Gun','Camera','Camcorder','SD Card'] 
}

with open('data/custom_data.yaml', 'w') as outfile:
    yaml.dump(data_yaml, outfile, default_flow_style=False)

print('done')


# YAML 파일 열기
with open('data/custom_data.yaml', 'r') as file:
    data = yaml.safe_load(file)

# YAML 파일 내용 출력
print(data)

# YOLOv5 학습
!python train.py --img 640 --batch 16 --epochs 3 --data /home/ec2-user/SageMaker/data/custom_data.yaml --weights ../yolov5_custom.pt --cache

### 파라미터 수정

import sagemaker
from sagemaker.tuner import HyperparameterTuner, IntegerParameter, ContinuousParameter
from sagemaker.pytorch import PyTorch
# SageMaker 세션 및 역할 설정
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# Estimator 설정
estimator = PyTorch(
    entry_point='train.py',
    source_dir='yolov5',  # YOLOv5 코드를 포함하는 디렉토리
    role=role,
    framework_version='1.9.0',
    py_version='py38',
    instance_count=1,
    instance_type='ml.m5.xlarge',
    hyperparameters={
        'img-size': 640,
    },
    sagemaker_session=sagemaker_session
)

# 튜닝 가능한 하이퍼파라미터 설정
hyperparameter_ranges = {
    'epochs': IntegerParameter(30, 100),
    'batch-size': IntegerParameter(8, 32),
    'learning-rate': ContinuousParameter(0.001, 0.01)
}

# 목표 메트릭 설정
objective_metric_name = 'mAP_0.5'
objective_type = 'Maximize'
metric_definitions = [
    {'Name': 'mAP', 'Regex': 'mAP@0.5: ([0-9\\.]+)'}
]

# 하이퍼파라미터 튜너 설정
tuner = HyperparameterTuner(
    estimator=estimator,
    objective_metric_name=objective_metric_name,
    hyperparameter_ranges=hyperparameter_ranges,
    metric_definitions=metric_definitions,
    max_jobs=20,
    max_parallel_jobs=2,
    objective_type=objective_type
)

# 데이터 경로 설정
bucket = 'asapv5-s3'
prefix = 'aws_dataset'
train_path = f's3://{bucket}/{prefix}/train'
val_path = f's3://{bucket}/{prefix}/val'

print('start tuning')
# 하이퍼파라미터 튜닝 시작
tuner.fit({'train': train_path, 'val': val_path})

# 최적의 하이퍼파라미터 조합 설정
best_hyperparameters = {
    'epochs': 70,
    'batch-size': 16,
    'learning-rate': 0.005
}

# Estimator 설정
final_estimator = sagemaker.estimator.Estimator(
    entry_point='train.py',
    source_dir='yolov5',  # YOLOv5 코드를 포함하는 디렉토리
    role=role,
    framework_version='1.9.0',
    py_version='py38',
    instance_count=1,
    instance_type='ml.p2.xlarge',
    hyperparameters=best_hyperparameters,
    sagemaker_session=sagemaker_session
)

# 최종 학습 시작
final_estimator.fit({'train': train_path, 'val': val_path})

import torch

# S3에서 모델 다운로드
s3.download_file(bucket_name, f'{s3_output_path}/yolov5_custom.pt', 'yolov5_custom.pt')

# 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5_custom.pt')

# 이미지 예측
img = 'path/to/your/test/image.jpg'  # 예측할 이미지 경로
results = model(img)

# 결과 출력
results.show()  # 화면에 예측 결과 표시
results.save()  # 결과를 파일로 저장

# test 데이터 성능 뽑기

import torch
import os
from tqdm import tqdm
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# S3 설정
bucket_name = 'asapv5-s3'
prefix_test_images = 'aws_dataset/test/images/'
prefix_test_labels = 'aws_dataset/test/labels/'

# 객체 리스트를 가져오는 함수
def list_s3_objects(bucket, prefix):
    object_list = []
    continuation_token = None
    while True:
        if continuation_token:
            response = s3.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix,
                ContinuationToken=continuation_token
            )
        else:
            response = s3.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix
            )
        
        if 'Contents' in response:
            for content in response['Contents']:
                object_list.append(content['Key'])
        
        if response.get('IsTruncated'):  # 더 많은 객체가 있을 경우
            continuation_token = response['NextContinuationToken']
        else:
            break
    return object_list

# 이미지 파일과 라벨 파일 목록 가져오기
test_image_list = list_s3_objects(bucket_name, prefix_test_images)
test_label_list = list_s3_objects(bucket_name, prefix_test_labels)

print(f"Total test images: {len(test_image_list)}")
print(f"Total test labels: {len(test_label_list)}")

# 테스트 데이터 다운로드
os.makedirs('data/test/images', exist_ok=True)
os.makedirs('data/test/labels', exist_ok=True)

def download_s3_files(bucket, file_list, local_dir):
    for file in file_list:
        local_path = os.path.join(local_dir, os.path.basename(file))
        s3.download_file(bucket, file, local_path)

download_s3_files(bucket_name, test_image_list, 'data/images/test')
download_s3_files(bucket_name, test_label_list, 'data/labels/test')

# 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5_custom.pt')

# 테스트 데이터 예측
predictions = []
targets = []

for img_path in tqdm(test_image_list):
    # 이미지 로드
    img = Image.open(f'data/test/images/{os.path.basename(img_path)}')
    
    # 모델 예측
    result = model(img)
    predictions.extend(result.xyxy[0].cpu().numpy())  # 예측 결과
    
    # 라벨 로드
    label_path = f'data/test/labels/{os.path.splitext(os.path.basename(img_path))[0]}.txt'
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            class_id = int(line[0])
            x_center = float(line[1])
            y_center = float(line[2])
            width = float(line[3])
            height = float(line[4])
            x1 = int((x_center - width/2) * img.width)
            y1 = int((y_center - height/2) * img.height)
            x2 = int((x_center + width/2) * img.width)
            y2 = int((y_center + height/2) * img.height)
            targets.append([class_id, x1, y1, x2, y2])

# 결과 시각화
def plot_image_with_boxes(image, boxes, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    ax.imshow(image)
    for box in boxes:
        class_id, x1, y1, x2, y2 = box
        box_width = x2 - x1
        box_height = y2 - y1
        rect = patches.Rectangle((x1, y1), box_width, box_height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1, f'class {class_id}', color='r', fontsize=8, verticalalignment='top')
    plt.show()

# 이미지와 예측 결과 시각화
for i in range(5):  # 처음 5개의 이미지만 시각화
    img_path = test_image_list[i]
    img = Image.open(f'data/test/images/{os.path.basename(img_path)}')
    boxes_pred = [pred[1:] for pred in predictions if int(pred[0]) == i]
    boxes_target = [target[1:] for target in targets if int(target[0]) == i]
    plot_image_with_boxes(img, boxes_pred, ax=None)
    plot_image_with_boxes(img, boxes_target, ax=None)


### 파인튜닝 (재학습)

import torch

# 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp/weights/best.pt')

# 새로운 이미지에 대해 예측 수행
img = 'path/to/your/new/image.jpg'  # 예측할 이미지 경로
results = model(img)

# 결과 출력 및 저장
results.show()  # 화면에 예측 결과 표시
results.save()  # 결과를 파일로 저장

# 감지 결과를 파일로 저장
detections = results.pandas().xyxy[0]  # 감지된 객체를 pandas DataFrame으로 가져오기

# 필요한 경우 데이터 가공
detections.to_csv('detections.csv', index=False)

class x_center y_center width height

# 새로운 데이터에 대한 yaml 파일 업데이트
data_yaml = {
    'train': 'path/to/your/updated/train/dataset',
    'val': 'path/to/your/updated/val/dataset',
    'nc': 42,  # 클래스 수 (사용자의 데이터에 맞게 수정)
    'names': ['object']  # 클래스 이름 (사용자의 데이터에 맞게 수정)
}

with open('data/updated_custom_data.yaml', 'w') as outfile:
    yaml.dump(data_yaml, outfile, default_flow_style=False)

# YOLOv5 모델 미세 조정
!python yolov5/train.py --img 640 --batch 16 --epochs 50 --data data/updated_custom_data.yaml --weights runs/train/exp/weights/best.pt --cache
