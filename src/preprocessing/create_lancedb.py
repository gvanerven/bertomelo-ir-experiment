import pandas
import lancedb
from lancedb.index import FTS
#from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

def create_table_from_parquet(parquet_path, field_name, table_name, db_path, model_id):
    df = pandas.read_parquet(parquet_path)
    #model = SentenceTransformer(model_id)
    #embeddings = model.encode(df[field_name].tolist(), batch_size=64, show_progress_bar=True)
    #df['embedding'] = embeddings.tolist() 
    
    db = lancedb.connect(db_path)
    table = db.create_table(table_name, df)
    if field_name in df.columns:
        table.create_index(field_name, config=FTS())
        
    return table

if __name__ == "__main__":
    list_models = [
        "unb-labia/BERTomelo-ModernBERT-Base-8k-Experimental",]
    list_datasets = [
        ("data/quati/passages.parquet", "passage", "quati_passages"),
        ("data/quati/queries.parquet", "query", "quati_queries"),
    ]
    db_path = "data/lancedb_quati"

    for parquet_path, field_name, table_name in list_datasets:
        for model_id in list_models:
            logger.info(f"Processing {parquet_path} with model {model_id}")
            create_table_from_parquet(
                parquet_path=parquet_path,
                field_name=field_name,
                table_name=table_name,
                db_path=db_path,
                model_id=model_id
    )