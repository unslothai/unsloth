from dataset_ingestion import (
    connect_to_s3,
    upload_dataset_to_s3,
    download_dataset_from_s3,
)


def ingest_dataset(dataset_path, bucket_name, aws_access_key_id, aws_secret_access_key):
    s3 = connect_to_s3(bucket_name, aws_access_key_id, aws_secret_access_key)
    if s3:
        upload_dataset_to_s3(s3, bucket_name, dataset_path)
    else:
        print("Failed to connect to S3 bucket")
