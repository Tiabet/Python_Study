import os
import json
import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm


# -------------------------------
# 1. Load client
# -------------------------------
def load_client():
    load_dotenv()
    api_key = os.getenv("ALICE_API_KEY")
    base_url = os.getenv("ALICE_EMB_URL")

    if not api_key or not base_url:
        raise ValueError("환경변수 ALICE_API_KEY 또는 ALICE_EMB_URL이 설정되지 않았습니다.")

    return OpenAI(api_key=api_key, base_url=base_url)


# -------------------------------
# 2. Batch Embedding Helper
# -------------------------------
def embed_texts(client, texts, model="text-embedding-3-small", batch_size=128):
    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch = texts[i: i + batch_size]
        res = client.embeddings.create(model=model, input=batch)
        for d in res.data:
            embeddings.append(d.embedding)

    return embeddings


# -------------------------------
# 3. Cosine similarity
# -------------------------------
def cosine(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# -------------------------------
# 4. Histogram visualization
# -------------------------------
def save_histogram(values, title, filename):
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=50, alpha=0.75, edgecolor='black')
    plt.title(title)
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.savefig(filename, dpi=150)
    plt.close()


# -------------------------------
# 5. Main logic with ranking
# -------------------------------
def main():
    client = load_client()

    print("Loading dataset...")
    with open("dataset/2wikimultihopqa.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # ---------------------------------------------
    # Collect all texts
    # ---------------------------------------------
    text_to_embedding = {}

    query_list = []
    answer_list = []

    # 새로: 전체 passage 저장 (unique)
    all_passages = set()

    # query별 gold supporting passages 저장
    gold_passages_per_query = []

    print("Collecting texts...")
    for item in tqdm(data, desc="Collect per sample"):
        q = item["question"]
        a = item["answer"]

        query_list.append(q)
        answer_list.append(a)

        passage_dict = {p[0]: p[1] for p in item["context"]}

        gold_passages = []

        for title, idx in item["supporting_facts"]:
            if title not in passage_dict:
                continue

            sentences = passage_dict[title]

            if idx < 0 or idx >= len(sentences):
                continue

            full_passage = " ".join(sentences)
            gold_passages.append(full_passage)
            all_passages.add(full_passage)

        gold_passages_per_query.append(gold_passages)

    # unique texts
    supporting_list = []
    for gps in gold_passages_per_query:
        supporting_list.extend(gps)

    all_texts = list(set(query_list + answer_list + list(all_passages)))

    print(f"Total unique texts to embed: {len(all_texts)}")

    # ---------------------------------------------
    # Embedding
    # ---------------------------------------------
    print("Starting embedding...")
    batch_embeddings = embed_texts(client, all_texts)

    for txt, emb in tqdm(zip(all_texts, batch_embeddings), total=len(all_texts)):
        text_to_embedding[txt] = emb

    # ---------------------------------------------
    # Ranking 계산 (query-level)
    # ---------------------------------------------
    print("Calculating ranking (per-query)...")

    passages_list = list(all_passages)
    embeddings_passages = [text_to_embedding[p] for p in passages_list]

    per_query_best_rank = []    # R_min(q)
    per_query_worst_rank = []   # R_max(q)
    per_query_mean_rank = []    # R_mean(q)

    per_query_mrr_any = []      # 1 / best_rank
    per_query_mrr_all = []      # 1 / worst_rank

    per_query_recall1_any = []
    per_query_recall5_any = []
    per_query_recall10_any = []

    per_query_recall1_all = []
    per_query_recall5_all = []
    per_query_recall10_all = []

    for q, gold_list in tqdm(zip(query_list, gold_passages_per_query),
                            total=len(query_list),
                            desc="Ranking per query"):

        if len(gold_list) == 0:
            continue

        q_emb = text_to_embedding[q]

        # 전체 passage에 대한 similarity 계산
        sims = np.array([cosine(q_emb, pe) for pe in embeddings_passages])

        # 이 query에서 gold passage들의 rank를 모두 모은다
        ranks_q = []

        for gold_p in gold_list:
            gold_emb = text_to_embedding[gold_p]
            gold_sim = cosine(q_emb, gold_emb)

            # rank = gold_sim 보다 큰 sim의 개수 + 1
            rank = int((sims > gold_sim).sum() + 1)
            ranks_q.append(rank)

        if not ranks_q:
            continue

        best_rank = min(ranks_q)
        worst_rank = max(ranks_q)
        mean_rank = float(np.mean(ranks_q))

        per_query_best_rank.append(best_rank)
        per_query_worst_rank.append(worst_rank)
        per_query_mean_rank.append(mean_rank)

        # MRR-like
        per_query_mrr_any.append(1.0 / best_rank)
        per_query_mrr_all.append(1.0 / worst_rank)

        # Recall@k (any / all)
        for k, any_list, all_list in [
            (1, per_query_recall1_any, per_query_recall1_all),
            (5, per_query_recall5_any, per_query_recall5_all),
            (10, per_query_recall10_any, per_query_recall10_all),
        ]:
            any_hit = 1.0 if best_rank <= k else 0.0
            all_hit = 1.0 if worst_rank <= k else 0.0
            any_list.append(any_hit)
            all_list.append(all_hit)



    # ---------------------------------------------
    # Final results
    # ---------------------------------------------
    print("\n===== FINAL QUERY-LEVEL RANK RESULTS =====")
    print(f"#queries evaluated: {len(per_query_best_rank)}")

    print(f"Best Rank  per query  (min gold rank)   - mean: {np.mean(per_query_best_rank):.2f}, median: {np.median(per_query_best_rank):.2f}")
    print(f"Worst Rank per query  (max gold rank)   - mean: {np.mean(per_query_worst_rank):.2f}, median: {np.median(per_query_worst_rank):.2f}")
    print(f"Mean Rank  per query  (avg gold rank)   - mean: {np.mean(per_query_mean_rank):.2f}, median: {np.median(per_query_mean_rank):.2f}")

    print(f"\nMRR_any (1 / best_rank): {np.mean(per_query_mrr_any):.4f}")
    print(f"MRR_all (1 / worst_rank): {np.mean(per_query_mrr_all):.4f}")

    print(f"\nRecall@1  (any): {np.mean(per_query_recall1_any):.4f}")
    print(f"Recall@5  (any): {np.mean(per_query_recall5_any):.4f}")
    print(f"Recall@10 (any): {np.mean(per_query_recall10_any):.4f}")

    print(f"\nRecall@1  (all): {np.mean(per_query_recall1_all):.4f}")
    print(f"Recall@5  (all): {np.mean(per_query_recall5_all):.4f}")
    print(f"Recall@10 (all): {np.mean(per_query_recall10_all):.4f}")


    # histogram
    # save_histogram(ranks, "Rank of Gold Supporting Passages", "hist_gold_passage_rank.png")


if __name__ == "__main__":
    main()
