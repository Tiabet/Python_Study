import json
import numpy as np
from tqdm import tqdm

# ====================================================
# 0. 원본 JSON 로딩 → 질문 타입 가져오기
# ====================================================
RAW_DATA_PATH = "dataset/2wikimultihopqa.json"
# RAW_DATA_PATH = "dataset/musique.json"
# RAW_DATA_PATH = "dataset/hotpot.jsonl"

with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# sample_idx → question_type
question_types = {i: sample["type"] for i, sample in enumerate(raw_data)}

# ====================================================
# 1. 임베딩 JSONL 로딩
# ====================================================
INPUT_PATH = "dataset/2wikimultihopqa_embeddings.jsonl"
# INPUT_PATH = "dataset/musique_embeddings.jsonl"
# INPUT_PATH = "dataset/hotpot_embeddings.jsonl"

queries = []
passages = []
passage_meta = []

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    for sample_idx, line in enumerate(f):
        record = json.loads(line)

        queries.append({
            "sample_idx": sample_idx,
            "embedding": np.array(record["query_embedding"], dtype=np.float32),
        })

        for p in record["passages"]:
            passages.append(np.array(p["embedding"], dtype=np.float32))
            passage_meta.append({
                "sample_idx": sample_idx,
                "title": p["title"],
                "is_gold": p["is_gold"]
            })

print(f"Loaded {len(queries)} queries")
print(f"Loaded {len(passages)} passages (global pool)")

queries_emb = np.stack([q["embedding"] for q in queries])
passages_emb = np.stack(passages)

# ====================================================
# 2. Cosine similarity 함수
# ====================================================
def cosine_similarity(a, b):
    a_norm = a / (np.linalg.norm(a) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return np.dot(b_norm, a_norm)


# ====================================================
# 3. Retrieval metrics 저장용
# ====================================================
metrics = {
    "global": {
        "avg_rank": [],
        "best_rank": [],
        "worst_rank": [],
        "mrr": [],
        "hit1": [],
        "hit5": [],
        "hit10": [],
    },
    "type": {}
}

# 타입 4개 할당
all_types = set(question_types.values())
for t in all_types:
    metrics["type"][t] = {
        "avg_rank": [],
        "best_rank": [],
        "worst_rank": [],
        "mrr": [],
        "hit1": [],
        "hit5": [],
        "hit10": [],
    }


# ====================================================
# 4. Query별 평가 수행
# ====================================================
for q_idx, q in tqdm(enumerate(queries), total=len(queries),
                     desc="Evaluating Queries"):

    q_emb = q["embedding"]
    sample_idx = q["sample_idx"]
    q_type = question_types[sample_idx]

    sims = cosine_similarity(q_emb, passages_emb)
    ranked = np.argsort(-sims)

    gold_idx = [
        i for i, meta in enumerate(passage_meta)
        if meta["sample_idx"] == sample_idx and meta["is_gold"]
    ]

    gold_ranks = [int(np.where(ranked == gi)[0][0]) + 1 for gi in gold_idx]

    best_rank = min(gold_ranks)
    worst_rank = max(gold_ranks)
    avg_rank = sum(gold_ranks) / len(gold_ranks)
    mrr = 1.0 / best_rank
    hit1 = 1 if best_rank <= 1 else 0
    hit5 = 1 if best_rank <= 5 else 0
    hit10 = 1 if best_rank <= 10 else 0

    # --- Global metrics 추가 ---
    for key, val in zip(
        ["avg_rank", "best_rank", "worst_rank", "mrr", "hit1", "hit5", "hit10"],
        [avg_rank, best_rank, worst_rank, mrr, hit1, hit5, hit10],
    ):
        metrics["global"][key].append(val)
        metrics["type"][q_type][key].append(val)


# ====================================================
# 5. 최종 결과 출력
# ====================================================
def summarize(title, values):
    return f"{title}: {np.mean(values):.3f}"

print("\n================== GLOBAL RESULTS ==================")
for key, values in metrics["global"].items():
    print(summarize(key, values))

print("\n================== TYPE RESULTS ====================")
for t in all_types:
    print(f"\n--- {t.upper()} ---")
    for key, values in metrics["type"][t].items():
        print(summarize(key, values))
