import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, TrainingArguments, PretrainedConfig
from datasets import load_dataset
from accelerate import Accelerator
import os
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as transforms

# 이미지 크기 통일 (512x512)
def resize_and_center_crop(image, target_size=512):
    transform = transforms.Compose([
        transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(target_size),  
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # [-1, 1] 정규화
    ])
    return transform(image)

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path, subfolder=subfolder
        )
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection
        return CLIPTextModelWithProjection.from_pretrained(
            pretrained_model_name_or_path, subfolder=subfolder
        )
    else:
        raise ValueError(f"{model_class} is not supported.")

# 1. 모델 로드 경로 설정
MODEL_PATH = "SG161222/Realistic_Vision_V6.0_B1_noVAE"
VAE_NAME = "stabilityai/sd-vae-ft-mse"

# 2. 개별 컴포넌트 로드하기
# UNet 로드 - variant 제거, torch_dtype 추가
unet = UNet2DConditionModel.from_pretrained(
    pretrained_model_name_or_path=MODEL_PATH,
    subfolder="unet",
    torch_dtype=torch.float16  # variant 대신 torch_dtype 사용
)

# VAE 로드
vae = AutoencoderKL.from_pretrained(
    pretrained_model_name_or_path=VAE_NAME,
    torch_dtype=torch.float16
)

# Tokenizer 로드
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=MODEL_PATH,
    subfolder="tokenizer",
    use_fast=False
)

# Text Encoder 로드
text_encoder = import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path=MODEL_PATH,
    subfolder="text_encoder"
)
text_encoder = text_encoder.to(torch.float16)

# 3. 파이프라인 생성
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_PATH,
    unet=unet,
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    torch_dtype=torch.float16
)
pipe.to("cuda")

# 4. 그래디언트 계산 비활성화
vae.requires_grad_(False)
text_encoder.requires_grad_(False)
unet.requires_grad_(False)

# 5. LoRA 설정
lora_config = LoraConfig(
    r=4,
    lora_alpha=4,
    lora_dropout=0.1,
    init_lora_weights="gaussian",
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
)

# 6. LoRA 모델 생성
peft_model = get_peft_model(unet, lora_config)
pipe.unet = peft_model

# 7. 데이터 로드
dataset = load_dataset("imagefolder", data_dir="/data/ephemeral/home/final_data", split="train")
print("dataset : ", dataset)

# 8. 캡션 파일 로드
captions_file = "/data/ephemeral/home/aaa/captions.txt"
captions = {}

with open(captions_file, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(":")
        if len(parts) == 2:
            filename, prompt = parts
            captions[filename.strip()] = prompt.strip()

# 9. 데이터 전처리 함수
def collate_fn(batch, target_size=512):
    images, filenames = [], []
    for item in batch:
        img = item["image"]
        filename = os.path.basename(img.filename) if hasattr(img, "filename") else "unknown.png"
        filenames.append(filename)
        img = resize_and_center_crop(img, target_size)
        images.append(img)

    return {
        "images": torch.stack(images).to("cuda").half(),
        "filenames": filenames
    }

dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda batch: collate_fn(batch, target_size=512))

# 10. 텍스트 처리 함수
def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    return text_inputs.input_ids

def encode_prompt(text_encoder, tokenizer, prompt):
    text_input_ids = tokenize_prompt(tokenizer, prompt).to("cuda")
    
    with torch.no_grad():
        text_embeddings = text_encoder(text_input_ids).last_hidden_state.to(dtype=torch.float16)
    
    return text_embeddings

# 11. 학습 설정
training_args = TrainingArguments(
    output_dir="./output/lora_weights",
    per_device_train_batch_size=1,
    num_train_epochs=10,
    save_steps=500,
    learning_rate=1e-5,  # 학습률 낮춤 (1e-4 -> 1e-5)
    remove_unused_columns=False,
    logging_dir="./logs",
)

# 12. 학습 실행
accelerator = Accelerator(mixed_precision="fp16")
optimizer = torch.optim.Adam(peft_model.parameters(), lr=training_args.learning_rate)

# 13. 그래디언트 클리핑 추가
max_grad_norm = 1.0

for epoch in range(training_args.num_train_epochs):
    for step, batch in enumerate(dataloader):
        img_tensors = batch["images"]
        filenames = batch["filenames"]
        prompts = [captions.get(fname, "A high-resolution portrait") for fname in filenames]

        # 디버깅 로그
        # if step % 10 == 0:
        #     print(f"파일명: {filenames}")
        #     print(f"매칭된 캡션: {prompts}")
        #     print("---")

        # 텍스트 임베딩 계산 - 수정된 부분: pipe의 인코더가 아닌 직접 로드한 인코더 사용
        text_embeddings = encode_prompt(text_encoder, tokenizer, prompts)

        # 입력 이미지 잠재 벡터 계산
        with torch.no_grad():
            latents = vae.encode(img_tensors).latent_dist.sample() * 0.18215
        
        # 노이즈 생성
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (img_tensors.shape[0],), device="cuda")
        
        # 노이즈 추가된 잠재 벡터 생성
        noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps) 
        
        # 텐서 값 확인 (디버깅용)
        if torch.isnan(noisy_latents).any() or torch.isinf(noisy_latents).any():
            print("Warning: NaN 또는 Inf가 noisy_latents에서 감지됨")
            continue

        # UNet 예측
        model_pred = pipe.unet(noisy_latents, timesteps, text_embeddings).sample    # noise를 얼마나 빼줘야 하는지
        
        # 예측값 확인 (디버깅용)
        if torch.isnan(model_pred).any() or torch.isinf(model_pred).any():
            print("Warning: NaN 또는 Inf가 model_pred에서 감지됨")
            continue

        # 손실 계산
        target = noise
        loss = torch.nn.functional.mse_loss(model_pred.float(), target.float())

        # 역전파
        accelerator.backward(loss)
        
        # 그래디언트 클리핑 추가
        torch.nn.utils.clip_grad_norm_(peft_model.parameters(), max_grad_norm)      #그래디언트 폭파 막기 위해
        
        optimizer.step()
        optimizer.zero_grad()                                                       #gradient 초기화
        
        print(f"Epoch {epoch + 1}, Step {step + 1}, Loss: {loss.item()}")

# 14. LoRA 가중치 저장
peft_model.save_pretrained("./output/lora_weights", safe_serialization=True)
print("✅ LoRA 학습 완료!")