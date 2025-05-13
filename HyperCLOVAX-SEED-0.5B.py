from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B"
).to(device)

tokenizer = AutoTokenizer.from_pretrained("naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B")

print(model)
# chat = [
#     {"role": "system",
#      "content": "- 당신은 온디바이스 LLM인 'CLOVA X'이며, 사용자와 짧은 대화를 통해 유용한 정보를 제공하는 역할입니다.\n- 대화의 흐름을 기억하고, 질문 간의 연관성을 파악해야 합니다.\n- 오늘은 2025년 4월 24일입니다."},
#
#     {"role": "user", "content": "지금 냉장고에 계란이랑 양파 밖에 없어."},
#     {"role": "assistant", "content": "계란과 양파만 있어도 오믈렛이나 계란말이를 만들 수 있어요. 둘 다 간단한 조리법이에요."},
#     {"role": "user", "content": "그럼 오믈렛 만드는 법 알려줘."}
# ]
#
# inputs = tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_dict=True, return_tensors="pt")
# inputs = {k: v.to(device) for k, v in inputs.items()}
#
# # stop_strings 제거하고 generate 실행
# output_ids = model.generate(**inputs, max_length=1024, stop_strings=["<|endofturn|>", "<|stop|>"], tokenizer=tokenizer)
#
# # 디코딩 및 답변만 추출
# decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0]
#
# # <|assistant|> 이후 텍스트만 가져오기
# if "<|assistant|>" in decoded:
#     answer = decoded.split("<|assistant|>")[-1].strip()
# else:
#     answer = decoded.strip()
#
# # 종료 문자열로 잘라내기
# for stop_str in ["<|endofturn|>", "<|stop|>"]:
#     if stop_str in answer:
#         answer = answer.split(stop_str)[0].strip()
#
# print(answer)
