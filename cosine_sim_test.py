import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv


def load_client():
    # .env 파일 읽기
    load_dotenv()

    api_key = os.getenv("ALICE_API_KEY")
    base_url = os.getenv("ALICE_EMB_URL")

    if not api_key or not base_url:
        raise ValueError("환경변수 ALICE_API_KEY 또는 ALICE_EMB_URL이 설정되지 않았습니다.")

    client = OpenAI(api_key=api_key, base_url=base_url)
    return client


def get_embedding(client, text, model="text-embedding-3-small"):
    response = client.embeddings.create(
        model=model,
        input=text
    )
    return response.data[0].embedding


def cosine_similarity(vec1, vec2):
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def main():
    # 🔽🔽 여기 문장만 바꿔서 테스트하면 됩니다 🔽🔽
    sent1 = "When did Lothair Ii's mother die?"
    sent2 = "20 March 851"
    # 🔼🔼 직접 수정 가능 🔼🔼

    client = load_client()

    print("문장 1:", sent1)
    print("문장 2:", sent2)

    emb1 = get_embedding(client, sent1)
    emb2 = get_embedding(client, sent2)

    sim = cosine_similarity(emb1, emb2)

    print("\n코사인 유사도:", sim)


if __name__ == "__main__":
    main()
