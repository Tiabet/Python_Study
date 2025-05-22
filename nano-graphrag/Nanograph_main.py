import json
from nano_graphrag import GraphRAG, QueryParam


def load_contexts(file_path: str) -> list[str]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def main():
    # 워킹 디렉토리와 사용할 모델 지정
    graph = GraphRAG(
        working_dir="./my_graph",
        llm_model="gpt-4o-mini",            # 답변 생성 모델
        embedding_model="text-embedding-3-small",  # 임베딩 모델
        chunk_size=1200,                       # 문서 청킹 크기
        chunk_overlap=100                      # 청킹 오버랩 크기
    )

    # contexts.txt에서 문서 읽어 삽입
    contexts = load_contexts("contexts.txt")
    for doc in contexts:
        graph.insert(doc)
    print(f"총 {len(contexts)}개의 컨텍스트를 그래프에 삽입했습니다.")

    # 대화형 질의 루프
    print("\n=== nano-graphrag 질의 모드 ===")
    print("질문을 입력하세요 ('exit' 입력 시 종료)")
    while True:
        q = input("질문: ")
        if q.lower() in ("exit", "quit"):
            print("종료합니다.")
            break
        answer = graph.query(
            q,
            param=QueryParam(mode="golbal")  # 'local', 'global', 'naive' 중 선택
        )
        print("답변:\n", answer)


if __name__ == "__main__":
    main()
