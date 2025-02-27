from diffusers import StableDiffusionPipeline
import torch

MODEL_PATH = "SG161222/Realistic_Vision_V6.0_B1_noVAE"

# 모델 로드
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16
)

# 포함된 컴포넌트 확인
print("✔ VAE:", pipe.vae is not None)
print("✔ Text Encoder:", pipe.text_encoder is not None)
print("✔ UNet:", pipe.unet is not None)
