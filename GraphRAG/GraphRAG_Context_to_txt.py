import json
import os
import tiktoken

# GPT-4o 기준 토크나이저 로드
encoding = tiktoken.encoding_for_model("gpt-4o")

domains = ["agriculture", "legal", "cs", "mix"]
input_dir = os.path.join("GraphRAG", "Ultradomain")

for domain in domains:
    input_path = os.path.join(input_dir, f"{domain}.jsonl")
    print(f"\n📂 Processing {input_path}...")

    seen_contexts = set()

    with open(input_path, "r", encoding="utf-8") as fin:
        for line in fin:
            obj = json.loads(line)
            context = obj.get("context", "").strip()
            if context:
                seen_contexts.add(context)

    token_lengths = [len(encoding.encode(context)) for context in seen_contexts]

    if token_lengths:
        min_len = min(token_lengths)
        max_len = max(token_lengths)
        avg_len = sum(token_lengths) / len(token_lengths)

        print(f"✅ Unique contexts: {len(seen_contexts)}")
        print(f"   🔹 Min tokens: {min_len}")
        print(f"   🔹 Max tokens: {max_len}")
        print(f"   🔹 Avg tokens: {avg_len:.2f}")
    else:
        print("⚠️ No valid context found.")
