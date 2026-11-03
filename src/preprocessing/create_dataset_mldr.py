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

def main(db_path, prefix_name, model_id, size=100000):    
    if not os.path.exists("data/dataset_mldr"):
        logger.info("Downloading dataset from Hugging Face...")
        train = load_dataset('Shitao/MLDR', 'pt', split='train', trust_remote_code=True)
        os.makedirs("data/dataset_mldr", exist_ok=True)
        train.to_parquet("data/dataset_mldr/train.parquet")
        train = pd.read_parquet("data/dataset_mldr/train.parquet")
    else:
        #train = load_dataset("parquet", data_files="data/dataset_mldr/train.parquet", split="train", keep_in_memory=False)
        #test = load_dataset("parquet", data_files="data/dataset_mldr/test.parquet", split="train", keep_in_memory=False)
        logger.info("Loading dataset from parquet files...")
        train = pd.read_parquet("data/dataset_mldr/train.parquet")
            
    #datasets = {'train': train, 'test': test}
    datasets = {'train': train}
    #model = SentenceTransformer(model_id)
    db = lancedb.connect(db_path)
    for split_name, dataset in datasets.items():
        if f"{split_name}_{prefix_name}_queries" not in db.list_tables():
            table_queries = db.create_table(f"{split_name}_{prefix_name}_queries", schema=pa.schema(
                [
                    pa.field("id", pa.int32()),
                    pa.field("query_id", pa.string()),
                    pa.field("query", pa.string()),
                    #pa.field("embedding", pa.list_(pa.float32(), model.get_embedding_dimension())),
                ])
            )
            
        if f"{split_name}_{prefix_name}_passages" not in db.list_tables():
            table_passages = db.create_table(f"{split_name}_{prefix_name}_passages", schema=pa.schema(
            [
                pa.field("id", pa.int32()),
                pa.field("passage_id", pa.string()),
                pa.field("passage", pa.string()),
                #pa.field("embedding", pa.list_(pa.float32(), model.get_embedding_dimension())),
            ]
            ))
            
        if f"{split_name}_{prefix_name}_qrels" not in db.list_tables():
            table_qrels = db.create_table(f"{split_name}_{prefix_name}_qrels", schema=pa.schema(
                [
                    pa.field("id_query", pa.int32()),
                    pa.field("id_passage", pa.int32()),
                    pa.field("score", pa.int32()),
                ]
            ))
        count_id_passage = 1
        record_query = []
        record_passages = []
        record_qrels = []
        idx_query = 1
        for _, row in dataset.iterrows():
            record_query = [{
                "id": idx_query,
                "query_id": row["query_id"],
                "query": row["query"],
            }]
            for passage in row["positive_passages"]:
                record_passages.append({
                    "id": count_id_passage,
                    "passage_id": passage['docid'],
                    "passage": passage['text'],
                })
                record_qrels.append({
                    "id_query": idx_query,
                    "id_passage": count_id_passage,
                    "score": 1,
                })
                count_id_passage += 1
                
            for passage in row["negative_passages"]:
                record_passages.append({
                    "id": count_id_passage,
                    "passage_id": passage['docid'],
                    "passage": passage['text'],
                })
                count_id_passage += 1

            idx_query += 1
            if idx_query % size == 0 or idx_query >= len(dataset):
                #logger.info(f"Processing rows {idx_query} of {len(dataset)} for split {split_name} and model {model_id}")
                logger.info(f"Processing rows {idx_query} of {len(dataset)} for split {split_name}")
                
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
        table_passages.create_index("passage", config=FTS())
        
if __name__ == "__main__":
    db_path = "data/lancedb_mldr"
    model_ids = ["unb-labia/BERTomelo-ModernBERT-Base-8k-Experimental"]
    for model_id in model_ids:
        #prefix_name = model_id.split("/")[-1]
        prefix_name = 'reranking'
        main(db_path=db_path, prefix_name=prefix_name, model_id=model_id, size=100000)
        
