from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
import torch
from PIL import Image
import os

# 이미지 경로
input_image_path = "/data/ephemeral/home/time_lapse/devil.png"
reference_image_path = "/data/ephemeral/home/time_lapse/reference.png"  # 참조 이미지 경로

# Realistic Vision 모델 경로
realistic_vision_model = "SG161222/Realistic_Vision_V6.0_B1_noVAE"

# ControlNet 모델 로드 (Reference 기반)
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_reference",
    torch_dtype=torch.float16
)

# Stable Diffusion + ControlNet 파이프라인 생성
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    realistic_vision_model,
    controlnet=controlnet,
    torch_dtype=torch.float16,
).to("cuda")

# 입력 이미지와 참조 이미지 로드
input_image = Image.open(input_image_path).convert("RGB")
reference_image = Image.open(reference_image_path).convert("RGB")

# 프롬프트 설정
prompt = "photo of a 30 years old person, realistic skin details"

# 이미지 생성
result = pipe(
    prompt=prompt,
    image=input_image,
    control_image=reference_image,  # 참조 이미지 입력
    num_inference_steps=50,
    guidance_scale=7.5
).images[0]

# 결과 저장
result.save("de_aged_result_reference.png")
print("생성된 이미지가 'de_aged_result_reference.png'에 저장되었습니다.")
