"""Evaluate a Hugging Face cross-encoder reranker on the Quati (pt-BR) IR dataset.

Pipeline: BM25 first-stage retrieval ("busca") followed by cross-encoder
reranking ("rerank") of the retrieved candidates, evaluated against Quati's
graded relevance judgments (qrels). Metrics are reported for the BM25-only
run and for the BM25+rerank run side by side, so the lift from reranking is
visible directly.

The reranker checkpoint (``--model_name_or_path``) can be either a local
directory (e.g. the ``--output_dir`` produced by ``train_rerank.py``) or a
model id on the Hugging Face Hub -- ``AutoModelForSequenceClassification``
resolves both transparently.

Dataset: ``unicamp-dl/quati`` requires ``trust_remote_code=True`` (custom
loading script) and ships several corpus-size configs (e.g. ``quati_1M``),
each with its own passages split and a matching ``<config>_qrels`` config,
plus a shared ``quati_all_topics`` config with the queries. Exact column and
split names have shifted across dataset versions, so this script resolves
them defensively from a list of likely candidates and logs what it picked.

Example
-------
python src/rerank/evaluation/quati_eval_rerank.py \\
    --model_name_or_path ./output/bertomelo-modernbert-rerank-ptbr \\
    --corpus_config quati_1M \\
    --top_k_retrieve 100 \\
    --k_values 1 5 10 20 100 \\
    --output_dir ./output/quati-eval
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def simple_tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BM25 retrieval + cross-encoder rerank evaluation on Quati.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Local directory or HF Hub id of the cross-encoder reranker to evaluate.",
    )
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--device", type=str, default=None, help="Defaults to cuda if available, else cpu.")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size used when scoring (query, passage) pairs.")

    # Dataset
    parser.add_argument("--dataset_name", type=str, default="unicamp-dl/quati")
    parser.add_argument(
        "--corpus_config",
        type=str,
        default="quati_1M",
        help="Quati corpus-size config, e.g. quati_1M. Its matching qrels config is '<corpus_config>_qrels'.",
    )
    parser.add_argument("--topics_config", type=str, default="quati_all_topics")
    parser.add_argument(
        "--trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Quati ships a custom dataset loading script and requires this to load.",
    )
    parser.add_argument(
        "--max_corpus_size",
        type=int,
        default=None,
        help="Subsample the corpus to this many passages (smoke tests only -- shrinks the recall/nDCG denominator's pool).",
    )
    parser.add_argument("--max_queries", type=int, default=None, help="Evaluate only this many judged queries.")

    # Retrieval / rerank
    parser.add_argument("--top_k_retrieve", type=int, default=100, help="BM25 candidates per query passed to the reranker.")
    parser.add_argument("--k_values", type=int, nargs="+", default=[1, 5, 10, 20, 100], help="Cutoffs for nDCG/Recall.")
    parser.add_argument("--mrr_k", type=int, default=10)
    parser.add_argument("--relevance_threshold", type=int, default=1, help="Minimum qrel value counted as 'relevant' for MAP/MRR/Recall.")

    parser.add_argument("--output_dir", type=str, default=None, help="If set, saves metrics.json and TREC-format run files here.")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def resolve_split(dataset_dict, preferred: Sequence[str], what: str):
    for name in preferred:
        if name in dataset_dict:
            logger.info("Using '%s' split for %s", name, what)
            return dataset_dict[name]
    if len(dataset_dict) == 1:
        name = next(iter(dataset_dict.keys()))
        logger.info("Using the only available split '%s' for %s", name, what)
        return dataset_dict[name]
    raise ValueError(
        f"Could not find a split for {what} among {list(preferred)}; "
        f"available splits: {list(dataset_dict.keys())}"
    )


def resolve_column(dataset, preferred: Sequence[str], what: str) -> str:
    for name in preferred:
        if name in dataset.column_names:
            logger.info("Using '%s' column for %s", name, what)
            return name
    raise ValueError(
        f"Could not find a column for {what} among {list(preferred)}; "
        f"available columns: {dataset.column_names}"
    )


def load_corpus(args) -> Tuple[List[str], List[str]]:
    logger.info("Loading Quati corpus (config=%s)", args.corpus_config)
    dd = load_dataset(args.dataset_name, args.corpus_config, trust_remote_code=args.trust_remote_code)
    corpus = resolve_split(dd, [f"{args.corpus_config}_passages", "passages", "corpus", "train"], "corpus")

    id_col = resolve_column(corpus, ["doc_id", "docid", "id", "_id", "pid"], "corpus doc id")
    text_col = resolve_column(corpus, ["passage", "text", "contents", "body", "document"], "corpus passage text")

    if args.max_corpus_size is not None and args.max_corpus_size < len(corpus):
        logger.warning(
            "Subsampling corpus to %d passages (out of %d) -- Recall/nDCG will not reflect the full collection.",
            args.max_corpus_size,
            len(corpus),
        )
        corpus = corpus.shuffle(seed=args.seed).select(range(args.max_corpus_size))

    doc_ids = [str(x) for x in corpus[id_col]]
    doc_texts = list(corpus[text_col])
    return doc_ids, doc_texts


def load_topics(args) -> Dict[str, str]:
    logger.info("Loading Quati topics (config=%s)", args.topics_config)
    dd = load_dataset(args.dataset_name, args.topics_config, trust_remote_code=args.trust_remote_code)
    topics = resolve_split(dd, [args.topics_config, "queries", "topics", "test", "train"], "topics")

    id_col = resolve_column(topics, ["query_id", "qid", "topic_id", "id", "_id"], "topic id")
    text_col = resolve_column(topics, ["query", "text", "title", "question"], "topic text")

    return {str(qid): text for qid, text in zip(topics[id_col], topics[text_col])}


def load_qrels(args) -> Dict[str, Dict[str, int]]:
    qrels_config = f"{args.corpus_config}_qrels"
    logger.info("Loading Quati qrels (config=%s)", qrels_config)
    dd = load_dataset(args.dataset_name, qrels_config, trust_remote_code=args.trust_remote_code)
    qrels_ds = resolve_split(dd, [qrels_config, "qrels", "test", "train"], "qrels")

    qid_col = resolve_column(qrels_ds, ["query_id", "qid", "topic_id"], "qrels query id")
    did_col = resolve_column(qrels_ds, ["doc_id", "docid", "pid", "id"], "qrels doc id")
    rel_col = resolve_column(qrels_ds, ["relevance", "rel", "label", "score", "judgement", "judgment"], "qrels relevance")

    qrels: Dict[str, Dict[str, int]] = {}
    for qid, did, rel in zip(qrels_ds[qid_col], qrels_ds[did_col], qrels_ds[rel_col]):
        qrels.setdefault(str(qid), {})[str(did)] = int(rel)
    return qrels


class CrossEncoderScorer:
    def __init__(self, model, tokenizer, device: str, max_length: int, batch_size: int):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size

    @torch.no_grad()
    def score(self, query: str, passages: List[str]) -> List[float]:
        scores: List[float] = []
        for i in range(0, len(passages), self.batch_size):
            batch = passages[i : i + self.batch_size]
            enc = self.tokenizer(
                [query] * len(batch),
                batch,
                padding=True,
                truncation="only_second",
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits.squeeze(-1)
            scores.extend(logits.float().cpu().tolist())
        return scores


def dcg_at_k(relevances: Sequence[float], k: int) -> float:
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances[:k]))


def ndcg_at_k(ranked_rels: Sequence[float], all_rels_for_query: Sequence[float], k: int) -> float:
    ideal = sorted(all_rels_for_query, reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(ranked_rels, k) / idcg


def average_precision(ranked_binary: Sequence[int], num_relevant: int) -> float:
    if num_relevant == 0:
        return 0.0
    hits = 0
    total = 0.0
    for i, rel in enumerate(ranked_binary):
        if rel:
            hits += 1
            total += hits / (i + 1)
    return total / num_relevant


def reciprocal_rank(ranked_binary: Sequence[int], k: int) -> float:
    for i, rel in enumerate(ranked_binary[:k]):
        if rel:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(ranked_binary: Sequence[int], num_relevant: int, k: int) -> float:
    if num_relevant == 0:
        return 0.0
    return sum(ranked_binary[:k]) / num_relevant


def evaluate_run(
    run: Dict[str, List[str]],
    qrels: Dict[str, Dict[str, int]],
    k_values: Sequence[int],
    mrr_k: int,
    relevance_threshold: int,
) -> Dict[str, float]:
    metrics = {f"ndcg@{k}": [] for k in k_values}
    metrics.update({f"recall@{k}": [] for k in k_values})
    metrics["map"] = []
    metrics[f"mrr@{mrr_k}"] = []

    for qid, ranked_doc_ids in run.items():
        judged = qrels.get(qid, {})
        all_rels = list(judged.values())
        num_relevant = sum(1 for r in all_rels if r >= relevance_threshold)

        ranked_rels = [judged.get(did, 0) for did in ranked_doc_ids]
        ranked_binary = [1 if r >= relevance_threshold else 0 for r in ranked_rels]

        for k in k_values:
            metrics[f"ndcg@{k}"].append(ndcg_at_k(ranked_rels, all_rels, k))
            metrics[f"recall@{k}"].append(recall_at_k(ranked_binary, num_relevant, k))
        metrics["map"].append(average_precision(ranked_binary, num_relevant))
        metrics[f"mrr@{mrr_k}"].append(reciprocal_rank(ranked_binary, mrr_k))

    return {name: float(np.mean(values)) if values else 0.0 for name, values in metrics.items()}


def save_trec_run(run: Dict[str, List[Tuple[str, float]]], run_name: str, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for qid, ranked in run.items():
            for rank, (doc_id, score) in enumerate(ranked, start=1):
                f.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} {run_name}\n")


def print_metrics_table(bm25_metrics: Dict[str, float], rerank_metrics: Dict[str, float]) -> None:
    header = f"{'metric':<12}{'bm25':>10}{'bm25+rerank':>14}"
    print(header)
    print("-" * len(header))
    for name in bm25_metrics:
        print(f"{name:<12}{bm25_metrics[name]:>10.4f}{rerank_metrics[name]:>14.4f}")


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
    args = parse_args()
    random.seed(args.seed)
    set_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading reranker from %s", args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, revision=args.revision, local_files_only=args.local_files_only
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path, revision=args.revision, local_files_only=args.local_files_only
    )
    model.to(device).eval()
    scorer = CrossEncoderScorer(model, tokenizer, device, args.max_length, args.batch_size)

    doc_ids, doc_texts = load_corpus(args)
    logger.info("Corpus size: %d passages", len(doc_ids))

    topics = load_topics(args)
    qrels = load_qrels(args)

    query_ids = [qid for qid in topics if qid in qrels]
    if not query_ids:
        raise ValueError("No topic has a matching entry in qrels -- check --corpus_config/--topics_config.")
    query_ids.sort()
    if args.max_queries is not None:
        query_ids = query_ids[: args.max_queries]
    logger.info("Evaluating %d judged queries", len(query_ids))

    logger.info("Tokenizing corpus and building BM25 index (this can take a while for large corpora)...")
    tokenized_corpus = [simple_tokenize(text) for text in tqdm(doc_texts, desc="Tokenizing corpus")]
    bm25 = BM25Okapi(tokenized_corpus)
    doc_ids_array = np.array(doc_ids)

    bm25_run: Dict[str, List[Tuple[str, float]]] = {}
    rerank_run: Dict[str, List[Tuple[str, float]]] = {}

    for qid in tqdm(query_ids, desc="Retrieving + reranking"):
        query_text = topics[qid]
        scores = bm25.get_scores(simple_tokenize(query_text))
        top_k = min(args.top_k_retrieve, len(scores))
        top_idx = np.argpartition(-scores, top_k - 1)[:top_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        candidate_ids = doc_ids_array[top_idx].tolist()
        candidate_scores = scores[top_idx].tolist()
        bm25_run[qid] = list(zip(candidate_ids, candidate_scores))

        candidate_texts = [doc_texts[i] for i in top_idx]
        rerank_scores = scorer.score(query_text, candidate_texts)
        order = np.argsort(rerank_scores)[::-1]
        rerank_run[qid] = [(candidate_ids[i], rerank_scores[i]) for i in order]

    bm25_run_ids = {qid: [did for did, _ in ranked] for qid, ranked in bm25_run.items()}
    rerank_run_ids = {qid: [did for did, _ in ranked] for qid, ranked in rerank_run.items()}

    bm25_metrics = evaluate_run(bm25_run_ids, qrels, args.k_values, args.mrr_k, args.relevance_threshold)
    rerank_metrics = evaluate_run(rerank_run_ids, qrels, args.k_values, args.mrr_k, args.relevance_threshold)

    print_metrics_table(bm25_metrics, rerank_metrics)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        save_trec_run(bm25_run, "bm25", os.path.join(args.output_dir, "bm25_run.trec"))
        save_trec_run(rerank_run, "rerank", os.path.join(args.output_dir, "rerank_run.trec"))
        with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump({"bm25": bm25_metrics, "bm25+rerank": rerank_metrics}, f, indent=2, ensure_ascii=False)
        logger.info("Saved run files and metrics.json to %s", args.output_dir)


if __name__ == "__main__":
    main()
