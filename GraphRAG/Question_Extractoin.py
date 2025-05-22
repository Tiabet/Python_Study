import json
import re
import os

# 변환할 도메인 목록
domains = ["agriculture", "legal", "cs", "mix"]

for domain in domains:
    input_path = f"questions/{domain}_questions.txt"
    output_path = f"questions/{domain}_questions.json"

    questions = []

    if not os.path.exists(input_path):
        print(f"[SKIP] {input_path} not found.")
        continue

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("- Question"):
                match = re.match(r"- Question \d+:\s*(.*)", line)
                if match:
                    question = match.group(1).strip()
                    questions.append({"query": question})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)

    print(f"[OK] {output_path} written with {len(questions)} questions.")
