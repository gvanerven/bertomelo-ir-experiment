import pandas as pd
import lancedb
from lancedb.index import FTS
#from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

def create_table_from_parquet(parquet_path, field_name, table_name, db_path):
    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded {len(df)} rows from {parquet_path} into DataFrame.")
    logger.info(f"Creating table {table_name} from DataFrame.")
    db = lancedb.connect(db_path)
    table = db.create_table(table_name, df)
    if field_name in df.columns:
        table.create_index(field_name, config=FTS())
        
    return table

if __name__ == "__main__":
    list_datasets = [
        ("data/quati/passages.parquet", "passage", "quati_passages"),
        ("data/quati/queries.parquet", "query", "quati_queries"),
        ("data/quati/rel_queries_passages.parquet", None, "quati_qrels")
    ]
    db_path = "data/lancedb_quati"

    for parquet_path, field_name, table_name in list_datasets:
        logger.info(f"Processing {parquet_path} with file {field_name} in {table_name}")
        create_table_from_parquet(
                parquet_path=parquet_path,
                field_name=field_name,
                table_name=table_name,
                db_path=db_path
        )