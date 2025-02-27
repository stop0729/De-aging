import os
import json
from glob import glob

# JSON 파일이 있는 경로
BASE_PATH = "/data/ephemeral/home/labeling_data"

def generate_prompts():
    all_json_files = glob(os.path.join(BASE_PATH, "VL_*", "*.json"))
    
    captions = []
    
    for json_file in all_json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        filename = data["filename"]
        age_past = data["age_past"]  # 촬영 당시 나이를 직접 사용!
        gender = data["gender"].lower()

        # 성별 지정 (어린 나이일 경우 boy/girl, 아니면 male/female 그대로 사용)
        child_gender = "boy" if gender == "male" else "girl"
        gender_label = child_gender if age_past < 10 else gender  # 10세 미만이면 boy/girl 사용
        
        # 캡션을 해당 이미지의 촬영 당시 나이로 설정
        prompt = f"photo of a {age_past}-year-old {gender_label}, detailed face, wrinkles, high-resolution portrait"
        
        captions.append(f"{filename}.png: {prompt}")

    return captions

if __name__ == "__main__":
    generated_captions = generate_prompts()

    # captions.txt로 저장
    with open("/data/ephemeral/home/aaa/captions.txt", "w", encoding="utf-8") as f:
        for caption in generated_captions:
            f.write(caption + "\n")
    
    print("====captions.txt 생성 완료!====")
