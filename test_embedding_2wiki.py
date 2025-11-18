import os
import json
import asyncio
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
# 2. Embed batch
# =============================
async def embed_batch(text_list, desc="Embedding"):
    embeddings = []
    batch_size = 256  # 원하는 batch size로 조절 가능

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
# 3. Load dataset
# =============================
DATA_PATH = "dataset/2wikimultihopqa.json"
OUTPUT_PATH = "dataset/2wikimultihopqa_embeddings.jsonl"

with open(DATA_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

print(f"Loaded {len(dataset)} samples.")


# =============================
# 4. Step 1 — 글로벌 리스트 구성
# =============================
query_texts = []
sf_texts = []
sf_map = []         # supporting fact 위치 정보
passage_texts = []
passage_map = []     # passage 위치 정보

for sample_idx, sample in enumerate(dataset):
    question = sample["question"]
    query_texts.append(question)

    # -------- Supporting Facts ---------
    supporting_facts = sample["supporting_facts"]
    context = sample["context"]

    # supporting fact sentence 추출
    for title, idx in supporting_facts:
        found = False
        for ctx_title, sentences in context:
            if ctx_title == title:
                if 0 <= idx < len(sentences):
                    sent = sentences[idx]
                    sf_texts.append(sent)
                    sf_map.append({
                        "sample_idx": sample_idx,
                        "sf_local_idx": len(sf_map)  # global index
                    })
                found = True
                break
        if not found:
            continue

    # gold passage title 집합
    gold_titles = {title for title, _ in supporting_facts}

    for ctx_title, sentences in context:
        passage_text = f"{ctx_title}. " + " ".join(sentences)
        passage_texts.append(passage_text)

        passage_map.append({
            "sample_idx": sample_idx,
            "title": ctx_title,
            "is_gold": ctx_title in gold_titles    # 🔥 gold 여부 저장
        })


# =============================
# 5. Step 2 — 글로벌 임베딩 3회 호출
# =============================
async def main():
    print("\n=== Embedding Queries ===")
    query_embeddings = await embed_batch(query_texts, desc="Query Embedding")

    print("\n=== Embedding Supporting Facts ===")
    sf_embeddings = await embed_batch(sf_texts, desc="Supporting Fact Embedding")

    print("\n=== Embedding Passages ===")
    passage_embeddings = await embed_batch(passage_texts, desc="Passage Embedding")

    # =============================
    # 6. Step 3 — sample별로 재조립
    # =============================
    output_samples = [{} for _ in range(len(dataset))]

    # 기본 틀 생성
    for i, sample in enumerate(dataset):
        output_samples[i] = {
            "query": sample["question"],
            "query_embedding": query_embeddings[i],

            "supporting_facts": [],
            "passages": []
        }

    # Supporting Facts 재조립
    sf_counter = 0
    for sf_idx, (text, emb) in enumerate(zip(sf_texts, sf_embeddings)):
        sample_idx = sf_map[sf_idx]["sample_idx"]
        output_samples[sample_idx]["supporting_facts"].append({
            "sentence": text,
            "embedding": emb
        })

    # Passage 재조립
    for p_idx, (text, emb) in enumerate(zip(passage_texts, passage_embeddings)):
        sample_idx = passage_map[p_idx]["sample_idx"]
        title = passage_map[p_idx]["title"]
        is_gold = passage_map[p_idx]["is_gold"]
        output_samples[sample_idx]["passages"].append({
            "title": title,
            "passage": text,
            "embedding": emb,
            "is_gold": is_gold
        })

    # =============================
    # 7. JSONL 저장
    # =============================
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for record in output_samples:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\nEmbedding Completed!")
    print(f"Saved to {OUTPUT_PATH}")


asyncio.run(main())
