"""Fine-tune a ModernBERT encoder as a cross-encoder reranker on mMARCO (Portuguese).

The script trains a sequence-classification head (a single relevance score,
``num_labels=1``) on top of a ModernBERT checkpoint using the ``train`` split
of ``unicamp-dl/mmarco`` (config ``portuguese``), which is made of
``(query, positive, negative)`` triples.

For every triple, the model scores ``(query, positive)`` and ``(query,
negative)`` independently and is optimized with a pairwise logistic loss
(``softplus(neg_score - pos_score)``), pushing the positive passage's score
above the negative one's -- the standard way to train a cross-encoder
reranker from triples.

Example
-------
python src/rerank/train_rerank.py \\
    --model_name_or_path unb-labia/BERTomelo-ModernBERT-Base-v1 \\
    --output_dir ./output/bertomelo-modernbert-rerank-ptbr \\
    --num_train_epochs 1 \\
    --per_device_train_batch_size 32 \\
    --max_train_samples 1000000

For the full ~39.7M-example train split, streaming avoids downloading the
whole dataset upfront:

python src/rerank/train_rerank.py --streaming --max_steps 200000
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import torch
import torch.nn.functional as F
from datasets import IterableDataset as HFIterableDataset
from datasets import load_dataset
from torch.utils.data import Dataset, IterableDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    set_seed,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune ModernBERT as a cross-encoder reranker on mMARCO (pt).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="unb-labia/BERTomelo-ModernBERT-Base-v1",
        help="Base ModernBERT checkpoint (encoder) to fine-tune.",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        help="Optional attention implementation override (e.g. 'sdpa', 'flash_attention_2').",
    )

    # Dataset
    parser.add_argument("--dataset_name", type=str, default="unicamp-dl/mmarco")
    parser.add_argument("--dataset_config", type=str, default="portuguese")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Stream the dataset instead of downloading it fully (recommended for the full train split).",
    )
    parser.add_argument(
        "--shuffle_buffer_size",
        type=int,
        default=10_000,
        help="Buffer size used to shuffle the dataset when --streaming is set.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Cap the number of training triples (useful for smoke tests / quick runs).",
    )
    parser.add_argument(
        "--eval_size",
        type=int,
        default=0,
        help="Number of triples held out from the train split for evaluation (0 disables eval).",
    )
    parser.add_argument("--max_length", type=int, default=256)

    # Optimization
    parser.add_argument("--output_dir", type=str, default="./output/modernbert-rerank-ptbr")
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Overrides num_train_epochs. Required when --streaming is set without --max_train_samples.",
    )
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")

    # Logging / checkpointing
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_strategy", type=str, default="steps", choices=["no", "steps", "epoch"])
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    # Hub
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    return parser.parse_args()


class RerankDataset(Dataset):
    """Map-style wrapper around a (query, positive, negative) triples dataset."""

    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        example = self.hf_dataset[idx]
        return {
            "query": example["query"],
            "positive": example["positive"],
            "negative": example["negative"],
        }


class RerankIterableDataset(IterableDataset):
    """Iterable wrapper around a streamed (query, positive, negative) triples dataset."""

    def __init__(self, hf_dataset: HFIterableDataset):
        self.hf_dataset = hf_dataset

    def __iter__(self) -> Iterable[Dict[str, str]]:
        for example in self.hf_dataset:
            yield {
                "query": example["query"],
                "positive": example["positive"],
                "negative": example["negative"],
            }


@dataclass
class RerankCollator:
    """Tokenizes (query, positive) and (query, negative) pairs separately."""

    tokenizer: PreTrainedTokenizerBase
    max_length: int = 256

    def __call__(self, features: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        queries = [f["query"] for f in features]
        positives = [f["positive"] for f in features]
        negatives = [f["negative"] for f in features]

        pos_enc = self.tokenizer(
            queries,
            positives,
            padding=True,
            truncation="only_second",
            max_length=self.max_length,
            return_tensors="pt",
        )
        neg_enc = self.tokenizer(
            queries,
            negatives,
            padding=True,
            truncation="only_second",
            max_length=self.max_length,
            return_tensors="pt",
        )

        batch: Dict[str, torch.Tensor] = {}
        for key, value in pos_enc.items():
            batch[f"pos_{key}"] = value
        for key, value in neg_enc.items():
            batch[f"neg_{key}"] = value
        return batch


class RerankTrainer(Trainer):
    """Trainer with a pairwise logistic loss over (positive, negative) scores."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        pos_inputs = {k[len("pos_"):]: v for k, v in inputs.items() if k.startswith("pos_")}
        neg_inputs = {k[len("neg_"):]: v for k, v in inputs.items() if k.startswith("neg_")}

        pos_logits = model(**pos_inputs).logits.squeeze(-1)
        neg_logits = model(**neg_inputs).logits.squeeze(-1)

        loss = F.softplus(neg_logits - pos_logits).mean()

        if return_outputs:
            logits = torch.stack([pos_logits, neg_logits], dim=1)
            return loss, {"logits": logits}
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        if prediction_loss_only:
            return (loss.detach(), None, None)
        return (loss.detach(), outputs["logits"].detach(), None)


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    logits = eval_pred.predictions
    pos_scores, neg_scores = logits[:, 0], logits[:, 1]
    accuracy = (pos_scores > neg_scores).mean()
    return {"pairwise_accuracy": float(accuracy)}


def build_datasets(args: argparse.Namespace):
    raw_dataset = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split=args.dataset_split,
        streaming=args.streaming,
    )

    if args.streaming:
        raw_dataset = raw_dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer_size)

        eval_raw = None
        train_raw = raw_dataset
        if args.eval_size > 0:
            eval_raw = raw_dataset.take(args.eval_size)
            train_raw = raw_dataset.skip(args.eval_size)
        if args.max_train_samples is not None:
            train_raw = train_raw.take(args.max_train_samples)

        train_dataset = RerankIterableDataset(train_raw)
        eval_dataset = RerankIterableDataset(eval_raw) if eval_raw is not None else None

        if args.max_steps <= 0 and args.max_train_samples is None:
            raise ValueError(
                "With --streaming you must set --max_steps, or bound the run with --max_train_samples."
            )
        return train_dataset, eval_dataset

    raw_dataset = raw_dataset.shuffle(seed=args.seed)

    eval_hf = None
    train_hf = raw_dataset
    if args.eval_size > 0:
        eval_hf = raw_dataset.select(range(args.eval_size))
        train_hf = raw_dataset.select(range(args.eval_size, len(raw_dataset)))
    if args.max_train_samples is not None:
        train_hf = train_hf.select(range(min(args.max_train_samples, len(train_hf))))

    train_dataset = RerankDataset(train_hf)
    eval_dataset = RerankDataset(eval_hf) if eval_hf is not None else None
    return train_dataset, eval_dataset


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
    args = parse_args()
    set_seed(args.seed)

    logger.info("Loading tokenizer and model from %s", args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model_kwargs: Dict[str, Any] = {"num_labels": 1}
    if args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path, **model_kwargs
    )

    logger.info(
        "Loading dataset %s/%s (split=%s, streaming=%s)",
        args.dataset_name,
        args.dataset_config,
        args.dataset_split,
        args.streaming,
    )
    train_dataset, eval_dataset = build_datasets(args)

    collator = RerankCollator(tokenizer=tokenizer, max_length=args.max_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=args.fp16,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )

    trainer = RerankTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics if eval_dataset is not None else None,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    logger.info("Saving final model to %s", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
