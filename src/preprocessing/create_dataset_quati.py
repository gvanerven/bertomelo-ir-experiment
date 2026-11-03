from datasets import load_dataset
import hashlib

def hash_passage(passages):
    hashs = [hashlib.sha256(passage.encode('utf-8')).hexdigest() for passage in passages['passage']]
    return {'passage_hash': hashs}

def split_passage(passages):
    split = [passage.split() for passage in passages['passage']]
    return {'split': split}


passages = load_dataset("unicamp-dl/quati", "quati_1M_passages", trust_remote_code=True)
qrels = load_dataset("unicamp-dl/quati", "quati_1M_qrels", trust_remote_code=True)
queries = load_dataset("unicamp-dl/quati", "quati_all_topics", trust_remote_code=True)

passages_hashed = passages["quati_1M_passages"].map(hash_passage, batched=True, batch_size=1000, num_proc=12)
passages_hashed = passages_hashed.map(split_passage, batched=True, batch_size=1000, num_proc=12)

passages_hashed.to_parquet("data/quati/passages.parquet")
qrels["quati_1M_qrels"].to_parquet("data/quati/rel_queries_passages.parquet")
queries["quati_all_topics"].to_parquet("data/quati/queries.parquet")


