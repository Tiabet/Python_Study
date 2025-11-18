import os
import json
import asyncio
import numpy as np
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI, AsyncOpenAI

from query_decomposition import decompose_query  # 네가 만든 모듈


# -------------------------------
# Load Clients
# -------------------------------
def load_embedding_client():
    load_dotenv()
    api_key = os.getenv("ALICE_API_KEY")
    base_url = os.getenv("ALICE_EMB_URL")
    if not api_key or not base_url:
        raise ValueError("환경변수 ALICE_API_KEY 또는 ALICE_EMB_URL이 설정되지 않았습니다.")
    return OpenAI(api_key=api_key, base_url=base_url)


def load_chat_client():
    load_dotenv()
    api_key = os.getenv("ALICE_OPENAI_KEY") or os.getenv("ALICE_API_KEY")
    base_url = os.getenv("ALICE_CHAT_URL")
    if not api_key or not base_url:
        raise ValueError("환경변수 ALICE_OPENAI_KEY/ALICE_API_KEY 또는 ALICE_CHAT_URL이 설정되지 않았습니다.")
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


# -------------------------------
# Batch Embedding
# -------------------------------
def embed_texts(client, texts, model="text-embedding-3-small", batch_size=128):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch = texts[i:i + batch_size]
        res = client.embeddings.create(model=model, input=batch)
        for d in res.data:
            embeddings.append(d.embedding)
    return embeddings


# -------------------------------
# Cosine similarity
# -------------------------------
def cosine(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# -------------------------------
# Histogram
# -------------------------------
def save_histogram(values, title, filename):
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=50, alpha=0.75, edgecolor="black")
    plt.title(title)
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.savefig(filename, dpi=150)
    plt.close()


# -------------------------------
# Async Decomposition Worker (100 parallel)
# -------------------------------
async def decompose_worker(client, question, sem):
    async with sem:
        try:
            result = await decompose_query(
                client,
                question,
                model="openai/gpt-4o-mini",
                temperature=0.1
            )
            return result
        except Exception as e:
            print(f"[ERROR] decompose_query 실패: {e}")
            return None


# -------------------------------
# Main
# -------------------------------
async def main():
    load_dotenv()
    chat_client = load_chat_client()
    emb_client = load_embedding_client()

    print("Loading dataset...")
    with open("dataset/2wikimultihopqa.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    num_questions = len(data)
    print(f"Total questions: {num_questions}")

    # ==============================
    # 1. 병렬 query decomposition
    # ==============================
    sem = asyncio.Semaphore(100)
    print("Decomposing queries (up to 100 concurrent)...")

    tasks = [
        asyncio.create_task(decompose_worker(chat_client, item["question"], sem))
        for item in data
    ]
    # gather는 순서를 보장
    decomp_results = await asyncio.gather(*tasks)

    # ==============================
    # 2. per-question 데이터 구조 채우기
    # ==============================
    per_q_subqueries = []        # 독립 subquery들 (depends_on == [])
    per_q_support_sents = []     # gold supporting sentences
    per_q_gold_passages = []     # gold supporting passages만
    per_q_all_passages = []      # 해당 question의 모든 passage들 (ranking 용)

    all_subqueries = []
    all_support_sents = []
    all_passages = []            # 모든 passage (gold + non-gold) for embedding

    print("Processing supporting facts + decompositions...")
    for i, item in enumerate(tqdm(data)):
        # --- context → passage dict ---
        passage_dict = {p[0]: p[1] for p in item["context"]}
        # question 전체 context passage (ranking 후보)
        all_passages_in_ctx = [" ".join(p[1]) for p in item["context"]]

        q_support_sents = []
        q_gold_passages = []

        # gold supporting facts
        for title, idx in item["supporting_facts"]:
            if title not in passage_dict:
                continue
            sentences = passage_dict[title]
            if idx < 0 or idx >= len(sentences):
                continue

            supp_sent = sentences[idx]
            full_passage = " ".join(sentences)

            q_support_sents.append(supp_sent)
            q_gold_passages.append(full_passage)

        per_q_support_sents.append(q_support_sents)
        per_q_gold_passages.append(q_gold_passages)
        per_q_all_passages.append(all_passages_in_ctx)

        all_support_sents.extend(q_support_sents)
        all_passages.extend(all_passages_in_ctx)

        # --- decomposition 결과 처리 ---
        result = decomp_results[i]
        if not result or not result.get("success", False):
            per_q_subqueries.append([])
            continue

        decomp = result["decomposition"]
        # depends_on이 비어 있는 SQ만 사용
        independent_subqs = [
            sq.question
            for sq in decomp.subquestions
            if not getattr(sq, "depends_on", None)  # [] 이면 False로 평가됨
        ]

        per_q_subqueries.append(independent_subqs)
        all_subqueries.extend(independent_subqs)

    # ==============================
    # 3. 임베딩 대상 텍스트 수집 & 중복 제거
    # ==============================
    all_texts = set(all_subqueries + all_support_sents + all_passages)
    all_texts = list(all_texts)
    print(f"Total unique texts to embed: {len(all_texts)}")

    print("Embedding all texts...")
    batch_embeddings = embed_texts(emb_client, all_texts)

    text_to_embedding = {}
    for txt, emb in tqdm(zip(all_texts, batch_embeddings), total=len(all_texts), desc="Storing embeddings"):
        text_to_embedding[txt] = emb

    # ==============================
    # 4. Similarity & Ranking 계산
    # ==============================
    print("Calculating similarities & ranking...")

    # subquery ↔ gold supporting sentence
    sims_subq_supp_all = []
    sims_subq_supp_max = []

    # subquery ↔ gold supporting passage
    sims_subq_goldpass_all = []
    sims_subq_goldpass_max = []

    # gold passage ranking (subq vs 모든 passage)
    gold_passage_ranks = []

    for i in tqdm(range(num_questions), desc="Per-question similarity"):
        subqs = per_q_subqueries[i]
        supp_sents = per_q_support_sents[i]
        gold_passages = per_q_gold_passages[i]
        all_pass_ctx = per_q_all_passages[i]

        if not subqs:
            continue

        # pre-embed 전체 context passages
        emb_ctx_passages = [text_to_embedding[p] for p in all_pass_ctx]

        for sq in subqs:
            emb_sq = text_to_embedding.get(sq)
            if emb_sq is None:
                continue

            # ---- subq vs gold supporting sentences ----
            if supp_sents:
                local_supp_sims = []
                for s in supp_sents:
                    emb_s = text_to_embedding.get(s)
                    if emb_s is None:
                        continue
                    sim = cosine(emb_sq, emb_s)
                    sims_subq_supp_all.append(sim)
                    local_supp_sims.append(sim)
                if local_supp_sims:
                    sims_subq_supp_max.append(max(local_supp_sims))

            # ---- subq vs gold supporting passages ----
            if gold_passages:
                local_goldpass_sims = []
                for gp in gold_passages:
                    emb_gp = text_to_embedding.get(gp)
                    if emb_gp is None:
                        continue
                    sim = cosine(emb_sq, emb_gp)
                    sims_subq_goldpass_all.append(sim)
                    local_goldpass_sims.append(sim)
                if local_goldpass_sims:
                    sims_subq_goldpass_max.append(max(local_goldpass_sims))

            # ---- ranking: SQ vs all context passages, gold passage 몇 등? ----
            if all_pass_ctx and gold_passages:
                # SQ vs all passages
                ctx_sims = [
                    cosine(emb_sq, emb_p) for emb_p in emb_ctx_passages
                ]
                # (score, idx) 로 정렬
                scored = list(enumerate(ctx_sims))  # (idx, score)
                scored.sort(key=lambda x: x[1], reverse=True)  # score desc

                # 각 gold passage에 대해 rank 계산
                for gp in gold_passages:
                    # gold passage가 context 리스트에서 어떤 index인지 찾기
                    indices_gp = [idx_p for idx_p, p in enumerate(all_pass_ctx) if p == gp]
                    if not indices_gp:
                        continue
                    gold_idx = indices_gp[0]

                    # rank 찾기
                    rank = None
                    for pos, (idx_p, _) in enumerate(scored, start=1):
                        if idx_p == gold_idx:
                            rank = pos
                            break
                    if rank is not None:
                        gold_passage_ranks.append(rank)

    # ==============================
    # 5. 결과 출력
    # ==============================
    print("\n===== FINAL RESULTS (INDEPENDENT SUBQs ONLY) =====")

    if sims_subq_supp_all:
        print(f"SubQ vs Gold Supporting Sentence (ALL pairs) mean: {np.mean(sims_subq_supp_all):.4f}")
    if sims_subq_supp_max:
        print(f"SubQ vs Gold Supporting Sentence (PER SubQ MAX) mean: {np.mean(sims_subq_supp_max):.4f}")

    if sims_subq_goldpass_all:
        print(f"SubQ vs Gold Supporting Passage (ALL pairs) mean: {np.mean(sims_subq_goldpass_all):.4f}")
    if sims_subq_goldpass_max:
        print(f"SubQ vs Gold Supporting Passage (PER SubQ MAX) mean: {np.mean(sims_subq_goldpass_max):.4f}")

    if gold_passage_ranks:
        print(f"\nGold Passage Rank 평균: {np.mean(gold_passage_ranks):.2f}")
        print(f"Gold Passage Rank 중앙값: {np.median(gold_passage_ranks):.2f}")

    # ==============================
    # 6. Debug samples
    # ==============================
    print("\n===== DEBUG SAMPLES (up to 3) =====")
    shown = 0
    for i in range(num_questions):
        if shown >= 3:
            break
        subqs = per_q_subqueries[i]
        supp_sents = per_q_support_sents[i]
        gold_passages = per_q_gold_passages[i]
        if not subqs or not supp_sents or not gold_passages:
            continue

        print(f"\n--- Sample {i+1} ---")
        print("Question:", data[i]["question"])
        print("Independent SubQs:")
        for sq in subqs:
            print("  -", sq)
        print("Gold Supporting Sentences:")
        for s in supp_sents:
            print("  -", s)
        print("Gold Passages (first 200 chars each):")
        for gp in gold_passages:
            print("  -", gp[:200], "...")

        sq0 = subqs[0]
        s0 = supp_sents[0]
        gp0 = gold_passages[0]
        sim_sq_s0 = cosine(text_to_embedding[sq0], text_to_embedding[s0])
        sim_sq_gp0 = cosine(text_to_embedding[sq0], text_to_embedding[gp0])
        print(f"Sim(SubQ0, GoldSupp0): {sim_sq_s0:.4f}")
        print(f"Sim(SubQ0, GoldPassage0): {sim_sq_gp0:.4f}")

        shown += 1

    # ==============================
    # 7. Histograms
    # ==============================
    print("\nSaving histograms...")

    if sims_subq_supp_all:
        save_histogram(sims_subq_supp_all,
                       "SubQ vs Gold Supporting Sentence (All pairs)",
                       "hist_subq_support_all_pairs.png")
    if sims_subq_supp_max:
        save_histogram(sims_subq_supp_max,
                       "SubQ vs Gold Supporting Sentence (Per SubQ max)",
                       "hist_subq_support_max_per_subq.png")

    if sims_subq_goldpass_all:
        save_histogram(sims_subq_goldpass_all,
                       "SubQ vs Gold Supporting Passage (All pairs)",
                       "hist_subq_goldpass_all_pairs.png")
    if sims_subq_goldpass_max:
        save_histogram(sims_subq_goldpass_max,
                       "SubQ vs Gold Supporting Passage (Per SubQ max)",
                       "hist_subq_goldpass_max_per_subq.png")

    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
