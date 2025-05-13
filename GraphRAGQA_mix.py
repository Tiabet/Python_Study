import json
import subprocess

# 설정
input_json = "questions/mix_questions.json"
output_json = "results/mix_results.json"
root_dir = "./graph_rag_project/rag_mix"
method = "local"  # or "global"

# 입력 불러오기
with open(input_json, "r", encoding="utf-8") as f:
    queries = json.load(f)

results = []

for i, item in enumerate(queries, 1):
    question = item["query"].strip()
    print(f"[{i}] Querying: {question}")

    try:
        response = subprocess.check_output(
            [
                "graphrag", "query",
                "--root", root_dir,
                "--method", method,
                "--query", question
            ],
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8"
        ).strip()

    except subprocess.CalledProcessError as e:
        response = f"[ERROR] {e.output.strip()}"

    results.append({
        "query": question,
        "result": response
    })

# JSON으로 저장
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n✅ All {len(results)} queries completed. Results saved to {output_json}")
