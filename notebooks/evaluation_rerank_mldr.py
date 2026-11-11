# %%
import lancedb
from ranx import Qrels, Run, evaluate, compare
from lancedb.query import MatchQuery
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import logging

import sys
sys.path.append("../src")
from rerank.rerankhf import rank

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger.info("Starting evaluation of reranking model on MLDR dataset...")
logger.info("Loading model and tokenizer...")
model_id = "cnmoro/Tucano160m-Reranker-EnPt"
model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=2
    )
tokenizer = AutoTokenizer.from_pretrained(model_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_length = 2048

# %%
logger.info("Connecting to LanceDB...")
db = lancedb.connect("../data/lancedb_mldr")

# %%
logger.info("Opening tables...")
tb_passages = db.open_table('train_reranking_passages')
tb_qrels = db.open_table('train_reranking_qrels')
tb_queries = db.open_table('train_reranking_queries')

# %%
qrels = tb_qrels.search().to_pandas()
qrels.sort_values(by="query_id", inplace=True)

def query_pandas_dict(df, query_col, doc_col, score_col):
    qid_atual = None
    q_docs = {}
    qrels_dict = {}
    for _, q in df.iterrows():
        if qid_atual is None:
            qid_atual = q[query_col]
            q_docs = {}

        if qid_atual != q[query_col]:
            qrels_dict[qid_atual] = q_docs
            qid_atual = q[query_col]
            q_docs = {}
        q_docs[q[doc_col]] = q[score_col]
    qrels_dict[qid_atual] = q_docs
    return qrels_dict

# %%
qrels_dict = query_pandas_dict(qrels, "query_id", "passage_id", "score")
qrels_dataset = Qrels(qrels_dict)
logger.info(f"Qrels loaded. Number of rows: {len(qrels)}")
# %%
MAX_RESULTS = 20
def get_results_dict(qrels_dict):
    results_dict = {}
    query_documents = {}
    for k in qrels_dict.keys(): # varrer banco para cada query_id e pegar os resultados do modelo
        q = tb_queries.search().where(f"query_id = '{k}'").limit(1).to_pandas() # recupera a query do banco de dados
        s = tb_passages.search(MatchQuery(q["query"].iloc[0], "passage"), query_type="fts").limit(MAX_RESULTS).to_pandas()[["passage_id", "passage", "_score"]]
        if not s.empty:
            query_documents[k] = (q["query"].iloc[0], s["passage"].tolist(), s["passage_id"].tolist())
            s["query_id"] = k
            qrels_dict = query_pandas_dict(s, "query_id", "passage_id", "_score")
            try:
                results_dict[k] = qrels_dict[k]
            except KeyError as e:
                print(f"Query {k} not found in results_dict: {e}")
    return results_dict, query_documents

logger.info("Retrieving results bm25...")
results_dict, query_documents = get_results_dict(qrels_dict)
logger.info(f"Number of results retrieved: {len(results_dict)}")
logger.info(f"Number of queries processed: {len(query_documents)}")
# -------------------------------------------------------------
# Avaliação de um modelo de recuperação de informação (BM25) usando a biblioteca ranx
bm25_dataset = Run(results_dict, name="bm25")


# %%
# -------------------------------------------------------------
# Avaliação BM25
# -------------------------------------------------------------
metrics = ["ndcg@20", "map@20", "mrr@20", "recall@20", "hit_rate@20"]
scores = evaluate(qrels_dataset, bm25_dataset, metrics)

print("Métricas BM25:")
for m, v in scores.items():
    print(f"  {m:<12}: {v:.4f}")


# %%
#Métricas Sistema Denso (BM25 max 100):
#  ndcg@20     : 0.8673
#  map@20      : 0.8443
#  mrr@20      : 0.8443
#  recall@20   : 0.9382
#  hit_rate@20 : 0.9382

result_reranked = {}
for query_id, (query, documents, doc_ids) in query_documents.items():
    print(f"Reranking query: {query_id}")
    ranked_results = rank(query, documents, tokenizer, model, device, normalize_scores=True)
    scores_docid = []
    for i in range(len(ranked_results)):
        scores_docid.append((doc_ids[i], ranked_results[i][1]))
    sorted_scores_docid = sorted(scores_docid, key=lambda x: x[1], reverse=True)
    result_reranked[query_id] = {x[0]: x[1] for x in sorted_scores_docid}
reranked_results_dataset = Run(result_reranked, name=f"reranked_{model_id.split('/')[-1]}")
metrics = ["ndcg@10", "map@10", "mrr@10", "recall@10", "hit_rate@10"]
scores = evaluate(qrels_dataset, reranked_results_dataset, metrics)

print("Métricas Reranking:")
for m, v in scores.items():
    print(f"  {m:<12}: {v:.4f}")

# %%
# -------------------------------------------------------------
# Comparação entre modelos com teste estatístico
# -------------------------------------------------------------
#report = compare(
#    qrels=qrels_dataset,
#    runs=[bm25_dataset, reranked_results_dataset],
#    metrics=["ndcg@20", "mrr@20", "recall@20"],
#    stat_test="student_t"  # Realiza teste t pareado
#)
#print("\nRelatório Comparativo:")
#print(report)


