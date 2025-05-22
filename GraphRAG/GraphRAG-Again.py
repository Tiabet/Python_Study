import json
import subprocess
import os


env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"

# 설정
input_json = "questions/agriculture_questions.json"
output_json = "results/agriculture_results_local.json"
root_dir = "./graph_rag_project/rag_agriculture"
method = "local"

# 질문 불러오기
with open(input_json, "r", encoding="utf-8") as f:
    queries = json.load(f)
query_map = {q["query"]: q for q in queries}

# 기존 결과 불러오기
with open(output_json, "r", encoding="utf-8") as f:
    existing_results = json.load(f)

updated_results = []
retry_count = 0

for i, item in enumerate(existing_results, 1):
    query = item["query"].strip()
    result = item["result"]

    if "traceback" in result.lower() or "[error]" in result.lower():
        print(f"[{i}] Retrying failed query: {query}")
        retry_count += 1
        try:
            response = subprocess.check_output(
                [
                    "graphrag", "query",
                    "--root", root_dir,
                    "--method", method,
                    "--query", query
                ],
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env
            ).strip()
        except subprocess.CalledProcessError as e:
            response = f"[ERROR] {e.output.strip()}"

        print(f"[{i}] Updated Response: {response}".encode("utf-8", errors="replace").decode("utf-8"))
        updated_results.append({
            "query": query,
            "result": response
        })
    else:
        updated_results.append(item)

# 덮어쓰기 저장
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(updated_results, f, ensure_ascii=False, indent=2)

print(f"\n✅ Retried {retry_count} queries and updated {output_json}. Total: {len(updated_results)}")
