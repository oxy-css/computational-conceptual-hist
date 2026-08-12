import os
from google.cloud import storage
from google.oauth2 import service_account
import json
import datasets


# Configuration
bucket_name = "project3102-model-bucket"
destination_blob_prefix = "Upload-Test/" # Folder path in GCS
local_dir = "./Upload-Test"

#works as intended, is just a little picky about input names
def upload_folder(credentials_path, bucket_name, destination_blob_prefix, local_dir):
    '''
    Upload all files in a folder to a GCS Bucket
        :param credentials_path: The path to a service account credential file in JSON format
        :param bucket_name: The bucket to place the folder in
        :param destination_blob_prefix: Folder path in the bucket to place the files inside INCLUDING THE NEW FOLDER (ex. old_folder/new_folder/)
        :param local_dir: Relative path of the local folder being uploaded
    '''
    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(bucket_name)

    # Upload each file in the local directory
    for root, _, files in os.walk(local_dir):
        for file in files:
            local_file_path = os.path.join(root, file)
            # Determine the relative path to maintain folder structure in GCS
            relative_path = os.path.relpath(local_file_path, local_dir)
            remote_path = os.path.join(destination_blob_prefix, relative_path)
            
            blob = bucket.blob(remote_path)
            blob.upload_from_filename(local_file_path, timeout=1000)
            print(f"Uploaded {local_file_path} to gs://{bucket_name}/{remote_path}")

    print("All model files uploaded to Google Cloud Storage.")

def upload_file(credentials_path, bucket_name, destination_blob_prefix, filepath):
    '''
    Upload all files in a folder to a GCS Bucket
        :param credentials_path: The path to a service account credential file in JSON format
        :param bucket_name: The bucket to place the folder in
        :param destination_blob_prefix: Folder path in the bucket to place the files inside INCLUDING THE NEW FOLDER (ex. old_folder/new_folder/)
        :param filepath: Relative path of the local file being uploaded
    '''

    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(bucket_name)
    #THE BLOB YOU CREATE NEEDS TO BE NAMED THE SAME AS THE FILE YOU UPLOAD
    blob_address = os.path.join(destination_blob_prefix, os.path.basename(filepath))
    blob = bucket.get_blob(blob_address)
    if blob == None:
        blob = bucket.blob(blob_address)
    blob.upload_from_filename("Semantics-Research/Upload-Test/test.txt", client=storage_client)
    print(f"Uploaded {filepath} to gs://{bucket_name}")

def download_file(credentials_path, bucket_name, file_blob_name, download_path):
    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(bucket_name)

    blob = bucket.get_blob(blob_name=file_blob_name)
    if blob==None:
        raise FileNotFoundError(
            f"The blob {file_blob_name} does not exist."
        )
    if os.path.dirname(download_path) != '':
        os.makedirs(os.path.dirname(download_path), exist_ok=True)
    blob.download_to_filename(download_path)
    print(f"Downloaded gs://{bucket_name}/{file_blob_name} → {download_path}")


def download_folder_from_bucket(credentials_path,
                                 bucket_name: str,
                                 source_prefix: str,
                                 destination_folder: str = ''):
    """
    Download an entire 'folder' (prefix) from a GCS bucket.

        :param credentials_path: filepath for service_account.json
        :param bucket_name: Name of the GCS bucket
        :param source_prefix: Folder path in bucket (e.g. 'models/my-model/')
        :param destination_folder: Local folder and filename to save content to
    """

    if not source_prefix.endswith("/"):
        source_prefix += "/"
    if destination_folder == '':
        destination_folder = source_prefix

    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    client = storage.Client(credentials=credentials)
    bucket = client.bucket(bucket_name)
    blobs = client.list_blobs(bucket_name, prefix=source_prefix)

    found = False
    for blob in blobs:
        # Skip directory placeholders
        if blob.name.endswith("/"):
            continue

        found = True

        # Remove the prefix from blob path
        relative_path = blob.name[len(source_prefix):]

        # Construct full local path
        local_path = os.path.join(destination_folder, relative_path)

        # Ensure directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Download file
        blob.download_to_filename(local_path)

        print(f"Downloaded gs://{bucket_name}/{blob.name} → {local_path}")

    if not found:
        raise FileNotFoundError(
            f"No objects found with prefix '{source_prefix}' in bucket '{bucket_name}'."
        )

    print("Folder download complete.")



def gcs_get_dataset(credentials_path, bucket_name, data_blob_path):
    '''
    retrieves a .jsonl file from GSC and formats it into a huggingface dataset object for use with huggingface models.
        :param credentials_path: filepath for service_account.json
        :param bucket_name: Name of the GCS bucket holding the data file
        :param data_blob_path: path to .jsonl blob to extract and format
    '''
    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(bucket_name)

    blob = bucket.get_blob(blob_name=data_blob_path)
    if blob==None:
        raise FileNotFoundError(
            f"The blob {data_blob_path} does not exist."
        )
    
    json_bytes = blob.download_as_bytes()
    json_data = json_bytes.decode('utf-8')
    data = [dict(json.loads(line)) for line in json_data.splitlines()]
    #ds = datasets.Dataset.from_dict(data)   
    ds = datasets.Dataset.from_list(data)
    return ds

def gcs_get_dataset_json_data(credentials_path, bucket_name, data_blob_path):
    '''
    retrieves a .jsonl file from GSC and formats it into a huggingface dataset object for use with huggingface models.
        :param credentials_path: filepath for service_account.json
        :param bucket_name: Name of the GCS bucket holding the data file
        :param data_blob_path: path to .jsonl blob to extract and format
    
    Note: this dataloading method reads the entire .jsonl file into memory 
    as a list of dictionaries, which is fine for small files but 
    not suitable for large datasets. 
    '''
    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(bucket_name)

    blob = bucket.get_blob(blob_name=data_blob_path)
    if blob==None:
        raise FileNotFoundError(
            f"The blob {data_blob_path} does not exist."
        )
    
    json_bytes = blob.download_as_bytes()
    json_data = json_bytes.decode('utf-8')
    data = [dict(json.loads(line)) for line in json_data.splitlines()]
    return data

# Configuration
if __name__ == "__main__":
    bucket_name = "project3102-model-bucket"
    destination_blob_prefix = "Upload-Test/" # Folder path in GCS
    local_dir = "Semantics-Research/Upload-Test"
    file = "Semantics-Research/Upload-Test/test.txt"
    service_account_path = "Semantics-Research/nlp-research-sp26.json"
    #upload_folder(credentials_path=service_account_path, bucket_name=bucket_name, destination_blob_prefix=destination_blob_prefix, local_dir=local_dir)
    #upload_file(credentials_path=service_account_path, bucket_name=bucket_name, destination_blob_prefix=destination_blob_prefix, filepath=file)
    #download_file(credentials_path=service_account_path, bucket_name=bucket_name, file_blob_name='Upload-Test/test.txt', download_path='test.txt')
    #download_folder_from_bucket(credentials_path=service_account_path, bucket_name=bucket_name, source_prefix='Upload-Test/', destination_folder='')
    #dataset = gcs_get_dataset(credentials_path=service_account_path, bucket_name="project3102-data-bucket", data_blob_path="sample_1950s_1960s.jsonl")
    dataset = gcs_get_dataset(credentials_path=service_account_path, bucket_name="project3102-data-bucket", data_blob_path="coha_sharded_full/train/1810s/shard_000.jsonl")
    print(dataset[0])


