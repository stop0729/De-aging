import torch
from diffusers import StableDiffusionXLPipeline
#from huggingface_hub import snapshot_download

# 모델 및 LoRA 파일 경로 설정
base_model = "stabilityai/stable-diffusion-xl-base-1.0"
lora_path = "/data/ephemeral/home/De-aging/IndianaJones.safetensors"  # 다운로드한 LoRA 파일 경로

# 모델 다운로드 (중단된 다운로드 자동 재개)
#snapshot_download(repo_id=base_model, resume_download=True)

# CUDA 11.8이므로 float16 사용하여 메모리 최적화
pipe = StableDiffusionXLPipeline.from_pretrained(
                                                pretrained_model_name_or_path = base_model,
                                                torch_dtype = torch.float16,
                                                variant = "fp16",
                                                use_safetensors = True,
                                                ).to("cuda")

# LoRA 가중치 적용
pipe.load_lora_weights(lora_path, use_safetensors=True)

# 프롬프트 입력 (Indiana Jones 스타일의 젊은 해리슨 포드)
prompt = "Harrison Ford, in his mid-70s, wearing a black coat with all his hair exposed and no hat."

# 이미지 생성
image = pipe(prompt, guidance_scale=10.0).images[0]

# 결과 이미지 저장 및 출력
output_path = "output_indiana_jones_3.png"
image.save(output_path)
print(f"이미지 생성 완료: {output_path}")
image.show()
