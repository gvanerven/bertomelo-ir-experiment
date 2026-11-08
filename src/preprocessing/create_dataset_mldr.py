from ctypes.util import test

from datasets import load_dataset
import hashlib
import lancedb
from lancedb.index import FTS
import logging
import pyarrow as pa
#from sentence_transformers import SentenceTransformer
import os
import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

def hash_passage(passages):
    hashs = [hashlib.sha256(passage.encode('utf-8')).hexdigest() for passage in passages['positive']]
    return {'positive_hash': hashs}

def main(db_path, prefix_name, num_rows_commitment=500):
    split_name = "train"
    if not os.path.exists("data/dataset_mldr"):
        logger.info("Downloading dataset from Hugging Face...")
        train = load_dataset('Shitao/MLDR', 'pt', split='train', trust_remote_code=True)
        os.makedirs("data/dataset_mldr", exist_ok=True)
        train.to_parquet("data/dataset_mldr/train.parquet")
        train = pd.read_parquet("data/dataset_mldr/train.parquet")
    else:
        logger.info("Dataset already downloaded. Loading from parquet files...")
        train = pd.read_parquet("data/dataset_mldr/train.parquet")
            
    logger.info(f"Dataset loaded. Number of rows in train: {len(train)}")

    db = lancedb.connect(db_path)
    if f"{split_name}_{prefix_name}_queries" not in db.list_tables():
        table_queries = db.create_table(f"{split_name}_{prefix_name}_queries", schema=pa.schema(
                [
                    pa.field("query_id", pa.string()),
                    pa.field("query", pa.string()),
                    #pa.field("embedding", pa.list_(pa.float32(), model.get_embedding_dimension())),
                ])
            )
        
    if f"{split_name}_{prefix_name}_passages" not in db.list_tables():
        table_passages = db.create_table(f"{split_name}_{prefix_name}_passages", schema=pa.schema(
        [
            pa.field("passage_id", pa.string()),
            pa.field("passage", pa.string()),
            #pa.field("embedding", pa.list_(pa.float32(), model.get_embedding_dimension())),
        ]
        ))
        
    if f"{split_name}_{prefix_name}_qrels" not in db.list_tables():
        table_qrels = db.create_table(f"{split_name}_{prefix_name}_qrels", schema=pa.schema(
            [
                pa.field("query_id", pa.string()),
                pa.field("passage_id", pa.string()),
                pa.field("score", pa.int32()),
            ]
        ))
    record_query = []
    record_passages = []
    record_qrels = []
    doc_ids = set()
    query_ids = set()
    idx_query = 1
    logger.info(f"Processing split train with {len(train)} rows")
    for _, row in train.iterrows():
        if row["query_id"] not in query_ids:
            record_query.append({
                "query_id": row["query_id"],
                "query": row["query"],
            })
            query_ids.add(row["query_id"])
        for passage in row["positive_passages"]:
            if passage['docid'] not in doc_ids:
                record_passages.append({
                    "passage_id": passage['docid'],
                    "passage": passage['text'],
                })
                doc_ids.add(passage['docid'])
                record_qrels.append({
                    "query_id": row["query_id"],
                    "passage_id": passage['docid'],
                    "score": 1,
                })
            
        for passage in row["negative_passages"]:
            if passage['docid'] not in doc_ids:
                record_passages.append({
                    "passage_id": passage['docid'],
                    "passage": passage['text'],
                })
                doc_ids.add(passage['docid'])

        idx_query += 1
        if idx_query % num_rows_commitment == 0:
            #logger.info(f"Processing rows {idx_query} of {len(dataset)} for split {split_name} and model {model_id}")
            logger.info(f"Processing rows {idx_query} of {len(train)} for split train")
            
            #with torch.no_grad():
            #aux = pd.DataFrame(record_query)
            #    aux['embedding'] = model.encode(aux['query'].tolist(), batch_size=size//2).tolist()
            #table_queries.add(aux)
            table_queries.add(record_query)
                                        
            #aux = pd.DataFrame(record_passages)
            #    aux['embedding'] = model.encode(aux['passage'].tolist(), batch_size=size//2).tolist()
            #table_passages.add(aux)
            table_passages.add(record_passages)
            
            table_qrels.add(record_qrels)
            
            record_query = []
            record_passages = []
            record_qrels = []
        #torch.cuda.empty_cache()

    logger.info(f"Finalizing processing for split train. Remaining rows: {len(record_query)}")
    table_queries.add(record_query)
    table_passages.add(record_passages)
    table_qrels.add(record_qrels)
    table_passages.create_index("passage", config=FTS())
        
if __name__ == "__main__":
    db_path = "data/lancedb_mldr"
    prefix_name = 'reranking'
    main(db_path=db_path, prefix_name=prefix_name, num_rows_commitment=500)
        
