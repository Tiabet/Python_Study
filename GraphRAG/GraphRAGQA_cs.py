import json
import subprocess

# 설정
input_json = "questions/cs_questions.json"
output_json = "results/cs_results.json"
root_dir = "./graph_rag_project/rag_cs"
method = "global"  # or "global"

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
        ).strip()

    except subprocess.CalledProcessError as e:
        response = f"[ERROR] {e.output.strip()}"


    print(f"[{i}] Response: {response}".encode("utf-8", errors="replace").decode("utf-8"))

    results.append({
        "query": question,
        "result": response
    })

        # 10번마다 자동 저장
    if i % 10 == 0:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[AUTO-SAVE] Saved {len(results)} results to {output_json}")

# JSON으로 저장
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n✅ All {len(results)} queries completed. Results saved to {output_json}")

# python GraphRAGQA_cs.py