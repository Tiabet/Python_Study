import os
import json
import asyncio
import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm

# =============================
# 1. Load env
# =============================
load_dotenv()
API_KEY = os.getenv("ALICE_API_KEY")
EMB_URL = os.getenv("ALICE_EMB_URL")

client = AsyncOpenAI(api_key=API_KEY, base_url=EMB_URL)

# =============================
# 2. Batch embedding
# =============================
async def embed_batch(text_list, desc="Embedding"):
    embeddings = []
    batch_size = 256

    for i in tqdm(range(0, len(text_list), batch_size), desc=desc):
        batch = text_list[i:i+batch_size]
        res = await client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        for item in res.data:
            embeddings.append(item.embedding)

    return embeddings


# =============================
# 3. Load HotpotQA JSONL
# =============================
INPUT_PATH = "dataset/hotpot.jsonl"
OUTPUT_PATH = "dataset/hotpot_embeddings.jsonl"

dataset = []
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        dataset.append(json.loads(line))

print(f"Loaded {len(dataset)} samples.")


# =============================
# 4. Global lists
# =============================
query_texts = []
sf_texts = []
sf_map = []

passage_texts = []
passage_map = []


for sid, sample in enumerate(dataset):
    # Query
    query_texts.append(sample["question"])

    # Supporting facts sentences
    for title, idx in sample["supporting_facts"]:
        # title 매칭
        for ctx_title, sentences in sample["context"]:
            if ctx_title == title:
                if idx < len(sentences):
                    sf_texts.append(sentences[idx])
                    sf_map.append({
                        "sample_idx": sid,
                        "title": title
                    })
                break

    # All passages
    for ctx_title, sentences in sample["context"]:
        passage = f"{ctx_title}. " + " ".join(sentences)
        passage_texts.append(passage)
        passage_map.append({
            "sample_idx": sid,
            "title": ctx_title
        })


# =============================
# 5. Embedding 호출
# =============================
async def main():
    print("\n=== Embedding Queries ===")
    query_emb = await embed_batch(query_texts, desc="Query Embedding")

    print("\n=== Embedding Supporting Facts ===")
    sf_emb = await embed_batch(sf_texts, desc="Supporting Fact Embedding")

    print("\n=== Embedding Passages ===")
    passage_emb = await embed_batch(passage_texts, desc="Passage Embedding")

    # ======================================
    # 6. Reassemble into per-sample JSONL
    # ======================================
    output = [{} for _ in range(len(dataset))]

    for i, sample in enumerate(dataset):
        output[i] = {
            "query": sample["question"],
            "query_embedding": query_emb[i],
            "supporting_facts": [],
            "passages": []
        }

    # Supporting facts
    for i, (sent, emb) in enumerate(zip(sf_texts, sf_emb)):
        sid = sf_map[i]["sample_idx"]
        title = sf_map[i]["title"]

        output[sid]["supporting_facts"].append({
            "title": title,
            "sentence": sent,
            "embedding": emb
        })

    # Passages
    for i, (text, emb) in enumerate(zip(passage_texts, passage_emb)):
        sid = passage_map[i]["sample_idx"]
        title = passage_map[i]["title"]

        # gold 여부 체크
        is_gold = any(p["title"] == title for p in output[sid]["supporting_facts"])

        output[sid]["passages"].append({
            "title": title,
            "passage": text,
            "embedding": emb,
            "is_gold": is_gold
        })

    # Save JSONL
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for rec in output:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("\n🔥 Embedding Complete.")
    print(f"Saved to {OUTPUT_PATH}")


asyncio.run(main())
