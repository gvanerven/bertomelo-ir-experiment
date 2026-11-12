import lancedb
from ranx import Qrels, Run, evaluate, compare
from lancedb.query import MatchQuery
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from math import ceil
import logging
import json

import sys
sys.path.append("../")
from rerankhf import rank

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S")

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

def get_results_dict(qrels_dict, tb_queries, tb_passages, max_results=10):
    results_dict = {}
    query_documents = {}
    for k in qrels_dict.keys(): # varrer banco para cada query_id e pegar os resultados do modelo
        q = tb_queries.search().where(f"query_id = '{k}'").limit(1).to_pandas() # recupera a query do banco de dados
        s = tb_passages.search(MatchQuery(q["query"].iloc[0], "passage"), query_type="fts").limit(max_results).to_pandas()[["passage_id", "passage", "_score"]]
        if not s.empty:
            query_documents[k] = (q["query"].iloc[0], s["passage"].tolist(), s["passage_id"].tolist())
            s["query_id"] = k
            qrels_dict = query_pandas_dict(s, "query_id", "passage_id", "_score")
            try:
                results_dict[k] = qrels_dict[k]
            except KeyError as e:
                print(f"Query {k} not found in results_dict: {e}")
    return results_dict, query_documents

def main(run_bm25_eval=False, query_leng_interval_idx=None, fert_leng_text_aprox=1, max_results=10):
    intervals = [(0, 512), (512, 1024), (1024, 2048), (2048, 4096), (4096, 100_000)]
    logger.info("Starting evaluation of reranking model on MLDR dataset...")
    model_id = "cnmoro/Tucano160m-Reranker-EnPt"
    logger.info(f"Loading model and tokenizer with model id: {model_id}...")
    model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=2
        )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_length = 2048

    logger.info("Connecting to LanceDB...")
    db = lancedb.connect("../../../data/lancedb_mldr")

    logger.info("Opening tables...")
    tb_passages = db.open_table('train_reranking_passages')
    tb_qrels = db.open_table('train_reranking_qrels')
    tb_queries = db.open_table('train_reranking_queries')

    if query_leng_interval_idx != None and query_leng_interval_idx >= 0 and query_leng_interval_idx < len(intervals):
        query_leng_interval_idx_text = f"{intervals[query_leng_interval_idx][0]}-{intervals[query_leng_interval_idx][1]}"
        qrels = tb_qrels.search().where(f"par_qtd_word_size > {ceil(abs(intervals[query_leng_interval_idx][0]/fert_leng_text_aprox))} and par_qtd_word_size <= {ceil(abs(intervals[query_leng_interval_idx][1]/fert_leng_text_aprox))}").to_pandas()
    else:
        query_leng_interval_idx_text = 'ALL_Dataset'
        qrels = tb_qrels.search().to_pandas()
    qrels.sort_values(by="query_id", inplace=True)

    # QRELs define the relevance of a passage to a query
    qrels_dict = query_pandas_dict(qrels, "query_id", "passage_id", "score")
    qrels_dataset = Qrels(qrels_dict)
    qrels_dataset.save(f"../../../data/evals/qresls_mldr_size{query_leng_interval_idx_text}_at{max_results}.json")
    logger.info(f"Qrels loaded. Number of rows: {len(qrels)}")

    logger.info("Retrieving results bm25...")
    results_dict, query_documents = get_results_dict(qrels_dict, tb_queries, tb_passages, max_results=max_results)
    logger.info(f"Number of results retrieved: {len(results_dict)}")
    logger.info(f"Number of queries processed: {len(query_documents)}")
    # -------------------------------------------------------------
    # Avaliação de um modelo de recuperação de informação (BM25) usando a biblioteca ranx
    if run_bm25_eval:
        bm25_dataset = Run(results_dict, name="bm25")
        bm25_dataset.save(f"../../../data/evals/bm25_mldr_run_size{query_leng_interval_idx_text}_at{max_results}.json")
        # -------------------------------------------------------------
        # Avaliação BM25
        # -------------------------------------------------------------
        metrics = ["ndcg@20", "map@20", "mrr@20", "recall@20", "hit_rate@20"]
        scores = evaluate(qrels_dataset, bm25_dataset, metrics, make_comparable=True)
        with open(f"../../../data/evals/bm25_mldr_scores_size{query_leng_interval_idx_text}_at{max_results}.json", "w") as f:
            json.dump(scores, f, indent=4)
        
        print(f"Métricas BM25: size {query_leng_interval_idx_text} at {max_results}")
        for m, v in scores.items():
            print(f"  {m:<12}: {v:.4f}")

        return bm25_dataset

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
    reranked_results_dataset.save(f"../../../data/evals/{model_id.split('/')[-1]}_mldr_run_size{query_leng_interval_idx_text}_at{max_results}.json")
    metrics = ["ndcg@10", "map@10", "mrr@10", "recall@10", "hit_rate@10"]
    scores = evaluate(qrels_dataset, reranked_results_dataset, metrics, make_comparable=True)
    with open(f"../../../data/evals/{model_id.split('/')[-1]}_mldr_scores_size{query_leng_interval_idx_text}_at{max_results}.json", "w") as f:
        json.dump(scores, f, indent=4)


    print(f"Métricas Reranking: size {query_leng_interval_idx_text} at {max_results} for {model_id.split('/')[-1]}")
    for m, v in scores.items():
        print(f"  {m:<12}: {v:.4f}")

    return scores

if __name__ == "__main__":
    main(run_bm25_eval=True, query_leng_interval_idx=4, max_results=10)
