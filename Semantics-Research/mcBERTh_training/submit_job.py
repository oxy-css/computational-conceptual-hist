from google.cloud import aiplatform
from google.oauth2 import service_account
import json

credentials_path = 'nlp-research-sp26.json'
credentials = service_account.Credentials.from_service_account_file(credentials_path)
with open(credentials_path) as f:
    service_account_email = json.load(f)['client_email']

a100 = {"machine_type": "a2-highgpu-1g", 
        "accelerator_type": "NVIDIA_TESLA_A100"}
l4 = {"machine_type": "g2-standard-8", 
      "accelerator_type": "NVIDIA_L4"}

tag = "macberth-adapt-v2" # retraining
# tag = "mcberth-decade-conditioned-v1" # trained
# tag = "bert-base-uncased-v1" # trained
# tag = "bert-base-uncased-conditioned-v1"

aiplatform.init(
    credentials=credentials,
    project="nlp-research-sp26",
    location="us-central1",
    staging_bucket="gs://project3102-model-bucket",
)

# Set the tensorboard instance name
tensorboard_name = "mcberth-tensorboard-non-conditioned-v2"
# tensorboard_name = "mcberth-tensorboard" # decade-conditioned
# tensorboard_name = "bert-tensorboard-non-conditioned"
# tensorboard_name = "bert-tensorboard-conditioned"
existing_tb = aiplatform.Tensorboard.list(filter=f'display_name="{tensorboard_name}"')
tb = existing_tb[0] if existing_tb else aiplatform.Tensorboard.create(display_name=tensorboard_name)
tensorboard_resource = tb.resource_name

job = aiplatform.CustomContainerTrainingJob(
    display_name=tag,
    container_uri=f"us-docker.pkg.dev/nlp-research-sp26/mcberth-training/mcberth-training:{tag}"
)

job.run(
    machine_type=l4["machine_type"],
    accelerator_type=l4["accelerator_type"],
    accelerator_count=1,
    replica_count=1,
    base_output_dir=f"gs://project3102-model-bucket/MacBERTh-domain-adaptation/{tag}",
    # base_output_dir=f"gs://project3102-model-bucket/BERT-domain-adaptation/{tag}",
    tensorboard=tensorboard_resource,
    service_account=service_account_email,
    sync=False,
)

print("Job submitted successfully!")