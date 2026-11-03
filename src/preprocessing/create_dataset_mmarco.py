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
    if not os.path.exists("data/dataset_mmarco"):
        dataset = load_dataset('unicamp-dl/mmarco', 'portuguese', split='train', trust_remote_code=True, keep_in_memory=False)
        train, test = dataset.train_test_split(test_size=0.20, seed=42).values()
        train.to_parquet("data/dataset_mmarco/train.parquet")
        test.to_parquet("data/dataset_mmarco/test.parquet")
        del dataset
    else:
        #train = load_dataset("parquet", data_files="data/dataset_mmarco/train.parquet", split="train", keep_in_memory=False)
        #test = load_dataset("parquet", data_files="data/dataset_mmarco/test.parquet", split="train", keep_in_memory=False)
        logger.info("Loading dataset from parquet files...")
        train = pd.read_parquet("data/dataset_mmarco/train.parquet")
        #test = pd.read_parquet("data/dataset_mmarco/test.parquet")
        
    #datasets = {'train': train, 'test': test}
    datasets = {'train': train}
    #model = SentenceTransformer(model_id)
    db = lancedb.connect(db_path)
    for split_name, dataset in datasets.items():
        if f"{split_name}_{prefix_name}_queries" not in db.list_tables():
            table_queries = db.create_table(f"{split_name}_{prefix_name}_queries", schema=pa.schema(
                [
                    pa.field("query_id", pa.int32()),
                    pa.field("query", pa.string()),
                    #pa.field("embedding", pa.list_(pa.float32(), model.get_embedding_dimension())),
                ])
            )
            
        if f"{split_name}_{prefix_name}_passages" not in db.list_tables():
            table_passages = db.create_table(f"{split_name}_{prefix_name}_passages", schema=pa.schema(
            [
                pa.field("passage_id", pa.int32()),
                pa.field("passage", pa.string()),
                #pa.field("embedding", pa.list_(pa.float32(), model.get_embedding_dimension())),
            ]
            ))
            
        if f"{split_name}_{prefix_name}_qrels" not in db.list_tables():
            table_qrels = db.create_table(f"{split_name}_{prefix_name}_qrels", schema=pa.schema(
                [
                    pa.field("query_id", pa.int32()),
                    pa.field("passage_id", pa.int32()),
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
                "query_id": idx_query,
                "query": row["query"],
            }]
            record_passages.append({
                "passage_id": count_id_passage,
                "passage": row["positive"],
            })
            record_passages.append({
                "passage_id": count_id_passage + 1,
                "passage": row["negative"],
            })
            record_qrels.append({
                "query_id": idx_query,
                "passage_id": count_id_passage,
                "score": 1,
            })
            count_id_passage += 2
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
    db_path = "data/lancedb_mmarco"
    model_ids = ["unb-labia/BERTomelo-ModernBERT-Base-8k-Experimental"]
    for model_id in model_ids:
        #prefix_name = model_id.split("/")[-1]
        prefix_name = 'reranking'
        main(db_path=db_path, prefix_name=prefix_name, model_id=model_id, size=100000)
        
