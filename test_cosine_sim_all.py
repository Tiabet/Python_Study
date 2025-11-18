import json
import numpy as np
from tqdm import tqdm

# ====================================================
# MusiQue 질문 타입 자동 추출 함수
# ====================================================
def detect_musique_type(sample):
    qid = sample.get("id", "")

    # 2-hop
    if qid.startswith("2hop"):
        return "2hop"

    # 3-hop variants
    if qid.startswith("3hop1"):
        return "3hop_1"
    if qid.startswith("3hop2"):
        return "3hop_2"

    # 4-hop variants
    if qid.startswith("4hop1"):
        return "4hop_1"
    if qid.startswith("4hop2"):
        return "4hop_2"
    if qid.startswith("4hop3"):
        return "4hop_3"

    return "unknown"



# ====================================================
# 원본 데이터 로딩 + 질문 타입 추출
# ====================================================
def load_raw_dataset(dataset_name):
    """
    dataset_name: "2wiki", "hotpot", "musique"
    """
    if dataset_name == "2wiki":
        path = "dataset/2wikimultihopqa.json"
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        question_types = {
            i: sample["type"]
            for i, sample in enumerate(raw)
        }

    elif dataset_name == "hotpot":
        path = "dataset/hotpot.jsonl"
        raw = []
        question_types = {}

        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                obj = json.loads(line)
                raw.append(obj)

                # Hotpot 질문 타입
                qtype = obj.get("type", "unknown")
                question_types[i] = qtype

    elif dataset_name == "musique":
        path = "dataset/musique.json"
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        question_types = {
            i: detect_musique_type(sample)
            for i, sample in enumerate(raw)
        }

    else:
        raise ValueError("dataset_name must be one of: 2wiki, hotpot, musique")

    return raw, question_types


# ====================================================
# 임베딩 로딩
# ====================================================
def load_embeddings(path):
    queries = []
    passages = []
    passage_meta = []

    with open(path, "r", encoding="utf-8") as f:
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

    return queries, passages, passage_meta


# ====================================================
# Cosine similarity
# ====================================================
def cosine_similarity(a, b):
    a_norm = a / (np.linalg.norm(a) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return np.dot(b_norm, a_norm)


# ====================================================
# 평가 스크립트 메인 함수
# ====================================================
def evaluate(dataset_name):
    print(f"=== Loading dataset: {dataset_name} ===")

    # 1) 원본 + 타입 로딩
    raw, question_types = load_raw_dataset(dataset_name)

    # 2) 임베딩 로딩
    emb_path = f"dataset/{dataset_name}_embeddings.jsonl"
    queries, passages, passage_meta = load_embeddings(emb_path)

    print(f"Loaded {len(queries)} queries")
    print(f"Loaded {len(passages)} passages")

    # numpy로 스택
    queries_emb = np.stack([q["embedding"] for q in queries])
    passages_emb = np.stack(passages)

    # metric 구조 준비
    all_types = set(question_types.values())

    metrics = {
        "global": {k: [] for k in
                   ["avg_rank", "best_rank", "worst_rank", "mrr", "hit1", "hit5", "hit10"]},
        "type": {
            t: {k: [] for k in
                ["avg_rank", "best_rank", "worst_rank", "mrr", "hit1", "hit5", "hit10"]}
            for t in all_types
        }
    }

    # 3) Query별 평가
    for q_idx, q in tqdm(enumerate(queries), total=len(queries)):
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

        # 저장
        for key, val in zip(
            ["avg_rank", "best_rank", "worst_rank", "mrr", "hit1", "hit5", "hit10"],
            [avg_rank, best_rank, worst_rank, mrr, hit1, hit5, hit10],
        ):
            metrics["global"][key].append(val)
            metrics["type"][q_type][key].append(val)

    # 출력
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


# ====================================================
# 실행 예시
# ====================================================
# evaluate("2wiki")
# evaluate("hotpot")
evaluate("musique")
