import json
import re

input_file = "results/agriculture_results_local.json"
output_file = "results/agriculture_results_local_cleaned.json"

def clean_result(text):
    # 1. ANSI escape 제거
    text = re.sub(r'\x1b\[[0-9;]*m', '', text)

    # 1. 인코딩 정보 제거 (예: cp949)
    text = re.sub(r"^\s*cp949\s*", "", text, flags=re.IGNORECASE)

    # 2. Graphrag 내부 설정 JSON 블록 제거
    text = re.sub(r'"default_vector_store": \{.*?\}\n?\}', '', text, flags=re.DOTALL)

    # 3. 로그 제거 (INFO, SUCCESS, DEBUG 등)
    text = "\n".join([
        line for line in text.splitlines()
        if not line.strip().startswith(("INFO:", "SUCCESS:", "DEBUG:", '"default_vector_store"'))
    ])

    match = re.search(r"Global Search Response:([\s\S]+)", text)
    if match:
        match.group(1).strip()  # 앞뒤 공백 제거
    else:
        text.strip()  # 혹시 매칭 안되면 원본 유지하되 공백만 제거

    # 4. [Data: ...] 제거
    text = re.sub(r"\[Data:.*?\]", "", text)

    # 5. 기타: 그래프 로그의 숫자 괄호 제거
    text = re.sub(r"\(\d+\)", "", text)

    # 6. 줄 바꿈 정리 + 앞뒤 공백
    return re.sub(r"\n{2,}", "\n\n", text).strip()


# 원본 로드 및 정제
with open(input_file, "r", encoding="utf-8") as fin:
    raw = json.load(fin)

cleaned = []
for item in raw:
    question = item["query"]
    raw_result = item["result"]
    cleaned_result = clean_result(raw_result)
    cleaned.append({
        "query": question,
        "result": cleaned_result
    })

# 결과 저장
with open(output_file, "w", encoding="utf-8") as fout:
    json.dump(cleaned, fout, ensure_ascii=False, indent=2)

print(f"✅ Cleaned {len(cleaned)} entries and saved to {output_file}")
