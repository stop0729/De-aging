import os
import json
from PIL import Image

def crop_images_based_on_labeling(labeling_data_dir, source_data_dir):
    """
    labeling_data 폴더 안의 각 폴더에 있는 JSON 파일의 box 좌표(x, y, w, h)를 기반으로
    source_data 폴더의 이름이 TL_XXXX와 일치하는 TS_XXXX 폴더 내 이미지 파일을 크롭하여
    'cropped' 하위 폴더에 저장합니다.
    """
    for folder_name in os.listdir(labeling_data_dir):
        if folder_name.startswith("VL_"):
            folder_number = folder_name.split("_")[1]  # 숫자 부분 추출
            corresponding_source_folder = f"VS_{folder_number}"
            
            labeling_folder_path = os.path.join(labeling_data_dir, folder_name)
            source_folder_path = os.path.join(source_data_dir, corresponding_source_folder)

            if os.path.isdir(labeling_folder_path) and os.path.isdir(source_folder_path):
                print(f"폴더 처리 중: {folder_name} -> {corresponding_source_folder}")

                # 'cropped' 하위 폴더 생성
                cropped_folder_path = os.path.join(source_folder_path, "cropped")
                os.makedirs(cropped_folder_path, exist_ok=True)

                for file in os.listdir(labeling_folder_path):
                    if file.endswith(".json"):
                        json_path = os.path.join(labeling_folder_path, file)
                        image_name = file.replace(".json", ".png")
                        image_path = os.path.join(source_folder_path, image_name)

                        if os.path.exists(image_path):
                            with open(json_path, "r", encoding="utf-8") as f:
                                data = json.load(f)

                            # box 좌표를 기반으로 이미지 크롭
                            for i, annotation in enumerate(data.get("annotation", [])):
                                box = annotation.get("box", {})
                                x = int(box.get("x", 0))
                                y = int(box.get("y", 0))
                                w = int(box.get("w", 0))
                                h = int(box.get("h", 0))

                                if w > 0 and h > 0:
                                    with Image.open(image_path) as img:
                                        cropped_img = img.crop((x, y, x + w, y + h))
                                        cropped_img_filename = f"{os.path.splitext(image_name)[0]}_crop_{i + 1}.png"
                                        cropped_img.save(os.path.join(cropped_folder_path, cropped_img_filename))
                                        print(f"저장됨: {cropped_img_filename} (좌표: x={x}, y={y}, w={w}, h={h})")
                        else:
                            print(f"이미지 파일 없음: {image_name} (경로: {image_path})")

    print("\n✅ 모든 이미지가 박스 좌표 기반으로 잘리고 'cropped' 폴더에 저장되었습니다.")

if __name__ == "__main__":
    LABELING_DATA_DIR = "labeling_data"
    SOURCE_DATA_DIR = "source_data"
    crop_images_based_on_labeling(LABELING_DATA_DIR, SOURCE_DATA_DIR)
