import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from PIL import Image

# 모델 이름
model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"

# 모델, 프로세서, 토크나이저 로드
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to("cuda")
preprocessor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 분석할 이미지 로드
image_path = "Test.jpg"  # 분석할 연예인 사진 경로
# image = Image.open(image_path).convert("RGB")

# 입력 구성
vlm_chat = [
        {"role": "system", "content": {"type": "text", "text": "System Prompt"}},
        {"role": "user", "content": {"type": "text", "text": "User Text 1"}},
        {
                "role": "user",
                "content": {
                        "type": "image",
                        "filename": "tradeoff_sota.png",
                        "image": "https://github.com/naver-ai/rdnet/blob/main/resources/images/tradeoff_sota.png?raw=true",
                        "ocr": "List the words in the image in raster order. Even if the word order feels unnatural for reading, the model will handle it as long as it follows raster order.",
                        "lens_keywords": "Gucci Ophidia, cross bag, Ophidia small, GG, Supreme shoulder bag",
                        "lens_local_keywords": "[0.07, 0.21, 0.92, 0.90] Gucci Ophidia",
                }
        },
        {
                "role": "user",
                "content": {
                        "type": "image",
                        "filename": "tradeoff.png",
                        "image": "https://github.com/naver-ai/rdnet/blob/main/resources/images/tradeoff.png?raw=true",
                }
        },
]

new_vlm_chat, all_images, is_video_list = preprocessor(vlm_chat)
preprocessed = preprocessor(all_images, is_video_list=is_video_list)
input_ids = tokenizer.apply_chat_template(
        new_vlm_chat, return_tensors="pt", tokenize=True, add_generation_prompt=True,
)

output_ids = model.generate(
        input_ids=input_ids.to(device="cuda"),
        max_new_tokens=8192,
        do_sample=True,
        top_p=0.6,
        temperature=0.5,
        repetition_penalty=1.0,
        **preprocessed,
)
print(tokenizer.batch_decode(output_ids)[0])