import boto3
from botocore.exceptions import NoCredentialsError

def connect_to_s3(bucket_name, aws_access_key_id, aws_secret_access_key):
    try:
        s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id,
                            aws_secret_access_key=aws_secret_access_key)
        return s3
    except NoCredentialsError:
        print("Credentials not available")
        return None

def upload_dataset_to_s3(s3, bucket_name, dataset_path):
    try:
        s3.upload_file(dataset_path, bucket_name, dataset_path)
        print("Dataset uploaded successfully")
    except FileNotFoundError:
        print("The file was not found")
    except NoCredentialsError:
        print("Credentials not available")

def download_dataset_from_s3(s3, bucket_name, dataset_path):
    try:
        s3.download_file(bucket_name, dataset_path, dataset_path)
        print("Dataset downloaded successfully")
    except FileNotFoundError:
        print("The file was not found")
    except NoCredentialsError:
        print("Credentials not available")