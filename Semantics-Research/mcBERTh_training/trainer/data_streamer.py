"""
This script builds a streaming dataset with decade-balancing and shuffling using Hugging face datasets.

Streaming, shuffled & interleaved;

data directory layout:
  coha_sharded_full/
    train/<decade>/shard_*.jsonl
    valid/<decade>/shard_*.jsonl
    
    (Only one shard per decade.)

Each JSONL line in our dataset includes the following fields: 

    {"file_name": "...", "doc_id": "...",  "decade": "...", 
    "genre": "...",  chunk_id: "...", "text": "...",}.

"""
import os
from datasets import interleave_datasets, load_dataset
from concurrent.futures import ThreadPoolExecutor
from google_cloud_save import download_file

LOCAL_CACHE = "/tmp/coha_shards"
BUCKET = "project3102-data-bucket"


DECADES = [
    "1810s", "1820s", "1830s", "1840s", "1850s", "1860s", "1870s", "1880s", "1890s", "1900s",
    "1910s", "1920s", "1930s", "1940s", "1950s", "1960s", "1970s", "1980s", "1990s", "2000s"
]

# Integer class id for each decade, used as the label for the Document Dating
DECADE_TO_ID = {d: i for i, d in enumerate(DECADES)}


def download_shards(service_account_path, root, split, decade):
    local_path = f"{LOCAL_CACHE}/{root}/{split}/{decade}/shard_000.jsonl"
    if not os.path.exists(local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        download_file(
            credentials_path=service_account_path,
            bucket_name=BUCKET,
            file_blob_name=f"{root}/{split}/{decade}/shard_000.jsonl",
            download_path=local_path,
        )
    return local_path

def build_decade_balanced_stream(
        service_account_path=None,
        root="coha_sharded_full",
        split="train",
        stream_buffer_size=10_000,
        interleave_buffer_size=50_000,
        seed=123,
        probabilities=None,
        stopping_strategy="all_exhausted",
        shuffle=True,
        use_decade_tokens=True):
    """
    Return:
        A mixed IterableDataset, a streaming iterable.
    """
    decades = [d for d in DECADES if not (split == 'valid' and d == '1810s')]

    # confirms decades are correct
    print(f"Data streamer: split={split}, {len(decades)} decades: {decades}")

    # Download all shards in parallel to local disk at startup
    with ThreadPoolExecutor(max_workers=8) as pool:
        local_paths = list(pool.map(
            lambda x: download_shards(service_account_path, root, split, x),
            decades
        ))

    # Check all shards were downloaded successfully
    print(f"Downloaded {len(local_paths)} shards:")
    for path in local_paths:
        print(f"  {path}")


    streams = []

    for local_path in local_paths:
        
        dataset = load_dataset(
            "json",
            data_files=[local_path],
            split="train",
            streaming=True, 
        )
        # Emit the decade as an integer class id (the Document Dating label),
        dataset = dataset.map(lambda x: {'dating_labels': DECADE_TO_ID[x['decade']]})

        if use_decade_tokens:
            #Train the Document Dating model with use_decade_tokens=False to prevent leakage
            dataset = dataset.map(lambda x: {
                'text': f'<decade_{str(x["decade"]).removesuffix("s")}> {x["text"]}'
            })
        dataset = dataset.select_columns(['text', 'dating_labels'])

        if shuffle:
            dataset = dataset.shuffle(buffer_size=stream_buffer_size, seed=seed)
        streams.append(dataset)

    mixed = interleave_datasets(streams, probabilities=probabilities,
                                seed=seed, stopping_strategy=stopping_strategy)

    if shuffle:
        mixed = mixed.shuffle(buffer_size=interleave_buffer_size, seed=seed)
    return mixed