import boto3
import json

s3_client = boto3.client('s3')

def lambda_handler(event, context):
    bucket_name = 'asapv5-s3'
    folder = 'processed_images/'
    
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix = folder)
    image_urls = []
    
    if 'Contents' in response:
        for obj in response['Contents']:
            if obj['Key'] != folder: 
                #image_url = f"https://asapv5-s3.s3.amazonaws.com/{obj['Key']}"
                image_url = s3_client.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': bucket_name, 'Key': obj['Key']},
                    ExpiresIn=3600  #3600초
                )
                image_urls.append(image_url)
    
    return {
        'statusCode': 200,
        'body': json.dumps({'image_urls': image_urls}),
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        }
    }
    