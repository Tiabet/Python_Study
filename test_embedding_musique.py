import os
import json
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm

# ==================================================
# 1. Environment
# ==================================================
load_dotenv()
API_KEY = os.getenv("ALICE_API_KEY")
EMB_URL = os.getenv("ALICE_EMB_URL")

client = AsyncOpenAI(api_key=API_KEY, base_url=EMB_URL)
MODEL = "text-embedding-3-small"
BATCH = 256


# ==================================================
# 2. Batch Embedding
# ==================================================
async def embed_batch(texts, desc):
    out = []
    for i in tqdm(range(0, len(texts), BATCH), desc=desc, smoothing=0):
        batch = texts[i:i+BATCH]
        res = await client.embeddings.create(
            model=MODEL,
            input=batch
        )
        for r in res.data:
            out.append(r.embedding)
    return out


# ==================================================
# 3. Load dataset
# ==================================================
DATA_PATH = "dataset/musique.json"
OUT_PATH = "dataset/musique_embeddings.jsonl"

dataset = json.load(open(DATA_PATH, "r", encoding="utf-8"))
print("Loaded", len(dataset), "Musique samples.")


# ==================================================
# 4. Build global lists
# ==================================================
query_texts = []
passage_texts = []
passage_map = []

for sample_idx, sample in enumerate(dataset):

    # -------- Query --------
    query_texts.append(sample["question"])

    # -------- Gold indices --------
    gold_indices = set()
    for qd in sample["question_decomposition"]:
        if "paragraph_support_idx" in qd:
            gold_indices.add(qd["paragraph_support_idx"])

    # -------- Passages --------
    for p in sample["paragraphs"]:
        p_idx = p["idx"]
        text = p["paragraph_text"]

        passage_texts.append(text)
        passage_map.append({
            "sample_idx": sample_idx,
            "title": p["title"],
            "p_idx": p_idx,
            "is_gold": (p_idx in gold_indices)
        })


# ==================================================
# 5. Embedding
# ==================================================
async def main():
    print("\n=== Embedding Queries ===")
    query_embs = await embed_batch(query_texts, "Query Embedding")

    print("\n=== Embedding Passages ===")
    passage_embs = await embed_batch(passage_texts, "Passage Embedding")

    # ==================================================
    # 6. Reassemble
    # ==================================================
    out_data = [{} for _ in range(len(dataset))]

    # Insert queries
    for i, sample in enumerate(dataset):
        out_data[i] = {
            "id": sample["id"],
            "question": sample["question"],
            "type": sample.get("type", ""),
            "query_embedding": query_embs[i],
            "passages": []
        }

    # Insert passages
    global_p_idx = 0
    for p_idx, (text, emb) in enumerate(zip(passage_texts, passage_embs)):
        meta = passage_map[p_idx]
        sidx = meta["sample_idx"]

        out_data[sidx]["passages"].append({
            "title": meta["title"],
            "passage": text,
            "p_idx": meta["p_idx"],
            "is_gold": meta["is_gold"],
            "embedding": emb
        })
        global_p_idx += 1

    # ==================================================
    # 7. Write JSONL
    # ==================================================
    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for row in out_data:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\nEmbedding Completed!")
    print("Saved to:", OUT_PATH)


asyncio.run(main())
