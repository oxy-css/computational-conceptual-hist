"""
Time-aware masked language model: joint MLM + Document Dating objective.

This adds BiTimeBERT's Document Dating (DD) task on top of the regular BERT
masked-language-modeling objective. A small classification head reads the final
[CLS] hidden state and predicts which decade the text chunk was written in. The
two objectives are trained jointly:

    loss = mlm_loss + dating_weight * dating_loss

Reference: Wang et al., "BiTimeBERT" (SIGIR 2023). We use ONLY their Document
Dating task (the [CLS] -> timestamp classifier), not TAMLM. Our MLM stays the
plain masked-LM we already run.

IMPORTANT — no time-token leakage:
Unlike the decade-token / TempoBERT model, this objective must run WITHOUT a
`<decade_XXXX>` token prepended to the text. If the decade is in the input,
predicting it is trivial and the backbone learns nothing. So train this model
with `use_decade_tokens=False` in the data streamer.

Contents:
  - TimeAwareMLMOutput          : model output carrying both losses + logits
  - BertForTimeAwareMLM         : BERT with the extra dating head
  - DataCollatorForMLMAndDating : MLM collator that also passes decade labels
"""
from dataclasses import dataclass
from typing import Optional, List

import torch
from torch import nn
from transformers import BertForMaskedLM, DataCollatorForLanguageModeling, Trainer
from transformers.utils import ModelOutput


@dataclass
class TimeAwareMLMOutput(ModelOutput):
    """Output type for BertForTimeAwareMLM.

    `loss` is the joint loss. `mlm_loss` / `dating_loss` are the components
    (populated during training, for logging). `dating_logits` are the decade
    logits; `logits` are the standard MLM logits.
    Fields left as None are omitted from the output, which keeps the tensors
    the Trainer gathers during evaluation small.
    """
    loss: Optional[torch.FloatTensor] = None
    mlm_loss: Optional[torch.FloatTensor] = None
    dating_loss: Optional[torch.FloatTensor] = None
    dating_logits: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None


class BertForTimeAwareMLM(BertForMaskedLM):
    """BERT trained with a joint MLM + Document Dating objective.

    num_decades   : number of time bins for the dating classifier (COHA: 20).
    dating_weight : lambda multiplier on the dating loss in the joint sum.

    Both values are read from `config` if not passed explicitly, so a model
    reloaded with `from_pretrained` keeps them automatically.
    """

    def __init__(self, config, num_decades: int = None, dating_weight: float = None):
        super().__init__(config)
        # Fall back to values stored on the config (set at first construction),
        # then to sensible defaults.
        num_decades = num_decades if num_decades is not None else getattr(config, "num_decades", 20)
        dating_weight = dating_weight if dating_weight is not None else getattr(config, "dating_weight", 1.0)

        self.num_decades = num_decades
        self.dating_weight = dating_weight
        # Persist on config so save_pretrained / from_pretrained round-trip them.
        self.config.num_decades = num_decades
        self.config.dating_weight = dating_weight

        # Document Dating head: h_[CLS] (hidden_size) -> decade logits (num_decades)
        self.dating_head = nn.Linear(config.hidden_size, num_decades)

        # Initialize the new head's weights the HF way.
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        labels=None,            # MLM labels produced by the data collator
        dating_labels=None,     # decade class id (int) per sequence
        **kwargs,               # absorb any extras Trainer may pass; not forwarded
    ):
        # 1) Run the BERT encoder + MLM head exactly as the base model does.
        #    Ask for hidden states so we can read the final [CLS] representation.
        mlm_out = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )
        mlm_loss = mlm_out.loss  # None if labels is None

        # 2) Document Dating: predict the decade from the final [CLS] state.
        cls_hidden = mlm_out.hidden_states[-1][:, 0]     # (batch, hidden_size)
        dating_logits = self.dating_head(cls_hidden)     # (batch, num_decades)

        dating_loss = None
        loss = mlm_loss
        if dating_labels is not None:
            dating_loss = nn.functional.cross_entropy(dating_logits, dating_labels)
            loss = (mlm_loss + self.dating_weight * dating_loss
                    if mlm_loss is not None else self.dating_weight * dating_loss)

        # During evaluation, return a minimal output so the Trainer only gathers
        # the small (batch, num_decades) dating logits — not the full-vocab MLM
        # logits or the hidden states. During training the extra fields are used
        # only for logging (the Trainer reads `.loss`).
        if not self.training:
            return TimeAwareMLMOutput(loss=loss, dating_logits=dating_logits)

        return TimeAwareMLMOutput(
            loss=loss,
            mlm_loss=mlm_loss,
            dating_loss=dating_loss,
            dating_logits=dating_logits,
            logits=mlm_out.logits,
        )


class DataCollatorForMLMAndDating(DataCollatorForLanguageModeling):
    """MLM collator that also carries the per-sequence decade label.

    The base collator only knows about MLM masking. We pop `dating_labels`
    before it runs (so tokenizer padding doesn't choke on the extra field),
    do the normal MLM collation, then stack the decade labels back on.
    """

    def torch_call(self, examples):
        dating = [ex.pop("dating_labels") for ex in examples]
        batch = super().torch_call(examples)  # standard MLM masking
        batch["dating_labels"] = torch.tensor(dating, dtype=torch.long)
        return batch


class TimeAwareTrainer(Trainer):
    """Trainer that additionally logs the MLM and Document Dating loss components.

    The model already computes the joint loss; here we just read the two
    components off the model output during training and average them into each
    logging step, so `metrics.jsonl` shows `mlm_loss` and `dating_loss` next to
    the combined `loss`. Everything else is the stock Trainer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._comp = {"mlm": 0.0, "dating": 0.0, "n": 0}

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        # Components are only populated in training mode (see the model's forward).
        if getattr(outputs, "mlm_loss", None) is not None and outputs.dating_loss is not None:
            self._comp["mlm"] += outputs.mlm_loss.item()
            self._comp["dating"] += outputs.dating_loss.item()
            self._comp["n"] += 1
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def log(self, logs, *args, **kwargs):
        # "loss" marks a training log event (eval events use "eval_loss").
        if "loss" in logs and self._comp["n"] > 0:
            n = self._comp["n"]
            logs["mlm_loss"] = round(self._comp["mlm"] / n, 4)
            logs["dating_loss"] = round(self._comp["dating"] / n, 4)
            self._comp = {"mlm": 0.0, "dating": 0.0, "n": 0}
        return super().log(logs, *args, **kwargs)


# ── Eval helpers (used by task.py) ────────────────────────────────────────────
def preprocess_logits_for_metrics(logits, labels):
    """Reduce eval logits to decade predictions before accumulation.

    In eval the model returns only the dating logits, so `logits` is that tensor
    (or a 1-tuple of it). Returning argmax keeps memory tiny across the eval set.
    """
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_dating_metrics(eval_pred):
    """Decade accuracy and mean-absolute-error (in decades) for the DD task."""
    preds, labels = eval_pred
    if isinstance(preds, (tuple, list)):
        preds = preds[0]
    preds = preds.reshape(-1)
    labels = labels.reshape(-1)
    acc = (preds == labels).mean()
    mae = abs(preds - labels).mean()
    return {"dating_accuracy": float(acc), "dating_mae_decades": float(mae)}
