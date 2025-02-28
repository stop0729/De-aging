from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from PIL import Image
import torch

# ControlNet 모델 로드 (Canny Edge 유지)
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-hed", torch_dtype=torch.float16) #diffusers/controlnet-canny-sdxl-1.0

base_model = "stabilityai/stable-diffusion-xl-base-1.0"
# Stable Diffusion + ControlNet 설정
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    pretrained_model_name_or_path = base_model, 
    controlnet=controlnet,
    torch_dtype=torch.float16,
    variant = "fp16",
)

# 모델을 float16 + CUDA에 적용
pipe.to("cuda", torch.float16)

image_path = "/data/ephemeral/home/De-aging/age_data/Training/source_data/TS_0001/0001_1992_29_00000074_D.png"
# 현재 얼굴 사진 로드
input_image = Image.open(image_path).convert("RGB")

# 프롬프트 설정
prompt = "photo of a 50 years old person, some wrinkles, photorealistic, ultra HD",

# ControlNet 변환 수행
young_face = pipe(prompt=prompt, image=input_image).images[0]

# 결과 저장
young_face.save("young_face_controlnet_2.jpg")
