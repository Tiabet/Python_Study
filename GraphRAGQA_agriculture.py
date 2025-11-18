import json
import subprocess
from tqdm import tqdm  # 1) tqdm 임포트

# 설정
input_json = "questions/agriculture_questions.json"
output_json = "results/agriculture_results_local_1.json"
root_dir = "./graph_rag_project/rag_agriculture"
method = "local"  # or "global"

# 입력 불러오기
with open(input_json, "r", encoding="utf-8") as f:
    queries = json.load(f)

results = []

# 2) tqdm으로 래핑: total=len(queries), desc에 표시할 문구 지정
for i, item in enumerate(
        tqdm(queries, total=len(queries), desc="Querying", unit="q"), 1):
    question = item["query"].strip()
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

    # tqdm이 알아서 라인 관리해 주기 때문에 print는 tqdm.write 사용 권장
    tqdm.write(f"[{i}] {question}\n→ {response}")

    results.append({
        "query": question,
        "result": response
    })

    # 10번마다 자동 저장
    if i % 10 == 0:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        tqdm.write(f"[AUTO-SAVE] Saved {len(results)} results to {output_json}")

# 최종 저장
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n✅ All {len(results)} queries completed. Results saved to {output_json}")
