# -*- coding: utf-8 -*-

import boto3

s3 = boto3.client('s3')
bucket_name = 'asapv5-s3'
model_artifact_path = 'yolov8s/artifact.tar.gz'

response = s3.head_object(Bucket=bucket_name, Key=model_artifact_path)
if response:
  print("Model artifact exists in S3.")
else:
  print("Model artifact does not exist in S3.")

import sagemaker
from sagemaker.pytorch import PyTorchModel

#generate SageMaker client
sagemaker_client = boto3.client('sagemaker')
role = 'arn:aws:iam::~:role/Asapv5-SageMakerFullAccess'

#Model artifact path
model_artifact = 's3://asapv5-s3/yolov8s/artifact.tar.gz'

#generate pytorch model
pytorch_model = PyTorchModel(
    model_data = model_artifact,
    role = role,
    entry_point = 'inference.py',
    framework_version = '1.7.1',
    py_version = 'py3'
)

#endpoint name
endpoint_name = 'asapv5-endpoint-9'

#Deploy endpoint
predictor = pytorch_model.deploy(
    initial_instance_count = 1,
    instance_type = 'ml.m5.large',
    endpoint_name = endpoint_name
)

print(f"Model deployed to endpoint : {predictor.endpoint_name}")