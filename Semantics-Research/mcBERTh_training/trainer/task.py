from transformers import (
    BertForMaskedLM, AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer, TrainingArguments, EarlyStoppingCallback,
    TrainerCallback
)
from data_streamer import build_decade_balanced_stream, DECADES
from datasets import Dataset as HFDataset
import os
import sys
import json
import math
import torch

# ── Hyperparameters ───────────────────────────────────────────────
model_name = "emanjavacas/MacBERTh"
# model_name = "bert-base-uncased"
epochs = 3
learning_rate = 5e-5
batch_size = 32
# gradient_accumulation_steps = (batchsize * 8) / (batchsize * #_GPU)
# 1 GPU:  256 / (32 * 1) = 8
# 2 GPUs: 256 / (32 * 2) = 4
gradient_accumulation_steps = 8
N = 2102849
max_steps = math.ceil(N / (batch_size * gradient_accumulation_steps)) * epochs
warmup_ratio = 0.05
weight_decay = 0.01
mlm_probability = 0.15
save_total_limit = 2
save_steps = 500
logging_steps = 100
early_stopping_patience = 5
early_stopping_threshold = 0.001

# Set to False if training a decade-conditioned model
use_decade_tokens = False
gcs_credentials = "nlp-research-sp26.json"

#temp parameters for quick testing
# max_steps = 8
# logging_steps = 2 
# save_steps = 2


# ── CUDA check ─────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on: {device.upper()}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("ERROR: CUDA not available. Aborting training job.")
    sys.exit(1)


# ── Tokenizer ─────────────────────────────────────────────────────
# Here are are mostly just using the tokenizer our model already uses. 
# However, were going add special tokens for each decade our data belongs to 
# this is how our model will differentiate words used in different time periods
# ex. <decade_1990>
print("Start script")

print('Building tokenizer')
def get_date_tokens(decades):
    return [f"<decade_{str(d).removesuffix('s')}>" for d in decades]


tokenizer = AutoTokenizer.from_pretrained(model_name)
if use_decade_tokens:
    tokenizer.add_special_tokens(
        {'additional_special_tokens': get_date_tokens(DECADES)})


def tokenize_data(examples):
    result = tokenizer(examples["text"], max_length=512, truncation=True)
    # word_ids are Python lists (not tensors) and not needed for MLM —
    # keeping them causes accelerate to hang when moving eval batches to GPU.
    return {k: v for k, v in result.items() if k != "word_ids"}


# ── Datasets ──────────────────────────────────────────────────────
# COHA requires a license so we can't store it locally — shards are downloaded
# from GCS to the VM's local disk at startup, then streamed during training.
# Make sure the service account JSON is present in the container.
print("building dataset")

train_dataset = build_decade_balanced_stream(
    service_account_path=gcs_credentials, use_decade_tokens=use_decade_tokens)

# No need to shuffle validation set
val_dataset = build_decade_balanced_stream(
    service_account_path=gcs_credentials, split='valid', shuffle=False,
    stopping_strategy="first_exhausted", use_decade_tokens=use_decade_tokens)

train_dataset = train_dataset.map(
    tokenize_data, batch_size=batch_size, batched=True, remove_columns=["text"])

print("Materializing validation set into memory...", flush=True)
val_dataset = val_dataset.map(
    tokenize_data, batch_size=batch_size, batched=True, remove_columns=["text"])
val_dataset = HFDataset.from_list(list(val_dataset))
print(f"Validation set ready: {len(val_dataset)} samples", flush=True)
print("dataset complete, formatting model")

# ── Model ─────────────────────────────────────────────────────────
# Defining our base model means downloading from huggingface and adding in our new decade tokens.
# we are also going to add in a callback function that helps it run smoother on google cloud
model = BertForMaskedLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability
)


class MetricsCallback(TrainerCallback):
    def __init__(self, output_dir, flush_every=10):
        self.metrics_path = f"{output_dir.rstrip('/')}/metrics.jsonl"
        self.flush_every = flush_every
        self.buffer = []

    def _flush(self):
        if not self.buffer:
            return
        with open(self.metrics_path, 'a') as f:
            f.write("\n".join(self.buffer) + "\n")
        self.buffer = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        # buffer the log entries in memory and only flush every N log events
        # to reduce latency in writing to disk
        self.buffer.append(json.dumps({"step": state.global_step, **logs}))
        if len(self.buffer) >= self.flush_every:
            self._flush()

    def on_train_end(self, *_, **kwargs):
        self._flush()


class ContiguousParamsCallback(TrainerCallback):
    def on_save(self, args, state, control, model=None, **kwargs):
        for name, param in model.named_parameters():
            if not param.data.is_contiguous():
                print(f"[WARNING] Non-contiguous tensor at save: {name}", flush=True)
                param.data = param.data.contiguous()
        return control

# ── Training ──────────────────────────────────────────────────────

print('Training start:', flush=True)

# Vertex AI sets AIP_MODEL_DIR automatically — use it as output dir

# However, it's necessary to map output_dir to Vertex AI's Cloud Storage FUSE Mount
# Vertex AI custom training automatically mounts GCS buckets inside the training container 
# using Cloud Storage FUSE at the /gcs/ path prefix.

aip_model_dir = os.environ.get("AIP_MODEL_DIR")
if aip_model_dir and aip_model_dir.startswith("gs://"):
    output_dir = aip_model_dir.replace("gs://", "/gcs/", 1)
    print(f"Translating GCS URI to FUSE path: {output_dir}", flush=True)
else:
    print(f"ERROR: AIP_MODEL_DIR not set or not a GCS path: {aip_model_dir}", flush=True)
    sys.exit(1)

aip_tb_log_dir = os.environ.get("AIP_TENSORBOARD_LOG_DIR")
if aip_tb_log_dir and aip_tb_log_dir.startswith("gs://"):
    logging_dir = aip_tb_log_dir.replace("gs://", "/gcs/", 1)
else:
    logging_dir = f"{output_dir.rstrip('/')}/logs"
print(f"Logging dir: {logging_dir}", flush=True)

print("Building TrainingArguments...", flush=True)
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    max_steps=max_steps,
    eval_strategy='steps',
    eval_steps=save_steps,
    save_strategy='steps',
    save_steps=save_steps,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    save_total_limit=save_total_limit,
    dataloader_drop_last=False,
    logging_dir=logging_dir,
    logging_steps=logging_steps,
    warmup_ratio=warmup_ratio,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    optim='adamw_torch',
)
print("TrainingArguments built.", flush=True)

print("Building Trainer...", flush=True)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(
        early_stopping_patience=early_stopping_patience,
        early_stopping_threshold=early_stopping_threshold),
        ContiguousParamsCallback(),
        MetricsCallback(output_dir)]
)
print("Trainer built.", flush=True)

# ── Save hyperparameters and training config ──────────────────────────────────────
params = {
    "model_name": model_name,
    "use_decade_tokens": use_decade_tokens,
    "training_corpus_size": N,
    "epochs_approximate": epochs,
    "max_steps": max_steps,
    "per_device_batch_size": batch_size,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "effective_batch_size": batch_size * gradient_accumulation_steps,
    "learning_rate": learning_rate,
    "warmup_ratio": warmup_ratio,
    "weight_decay": weight_decay,
    "optimizer": "adamw_torch",
    "mlm_probability": mlm_probability,
    "eval_steps": save_steps,
    "save_steps": save_steps,
    "save_total_limit": save_total_limit,
    "early_stopping_patience": early_stopping_patience,
    "early_stopping_threshold": early_stopping_threshold,
}
params_path = f"{output_dir.rstrip('/')}/hyperparameters.json"
with open(params_path, 'w') as f:
    json.dump(params, f, indent=4)
print(f"Hyperparameters and training config saved to {params_path}", flush=True)

# ── Train ─────────────────────────────────────────────────────────
print("starting training loop", flush=True)
# Ensure all params are contiguous before training starts — non-contiguous
# tensors cause safetensors to crash on checkpoint saves mid-training.
for param in model.parameters():
    param.data = param.data.contiguous()

trainer.train(resume_from_checkpoint=False)
print("training complete")

# ── Save best model ────────────────────────────────────────────────────
print("saving model")
save_path = f"{output_dir.rstrip('/')}/best"
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)
print(f"Model saved to {save_path}")


