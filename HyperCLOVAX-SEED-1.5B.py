from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
).to(device)

tokenizer = AutoTokenizer.from_pretrained("naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B")

chat = [
  {"role": "tool_list", "content": ""},
  {"role": "system", "content": "- AI 언어모델의 이름은 \"CLOVA X\" 이며 네이버에서 만들었다.\n- 오늘은 2025년 04월 24일(목)이다."},
  {"role": "user", "content": "역사상 가장 재미있는 영화 5편 추천해줘. 영화 제목, 영화 감독, 영화 줄거리, 출연 배우 순으로 정리해주고, 답변 포맷을 jsonl 파일 형식으로 해줘."},
]

inputs = tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_dict=True, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

# stop_strings 제거하고 generate 실행
output_ids = model.generate(**inputs, max_length=1024, stop_strings=["<|endofturn|>", "<|stop|>"], tokenizer=tokenizer)


# 디코딩 및 답변만 추출
decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0]

# <|assistant|> 이후 텍스트만 가져오기
if "<|assistant|>" in decoded:
    answer = decoded.split("<|assistant|>")[-1].strip()
else:
    answer = decoded.strip()

# 종료 문자열로 잘라내기
for stop_str in ["<|endofturn|>", "<|stop|>"]:
    if stop_str in answer:
        answer = answer.split(stop_str)[0].strip()

print(answer)