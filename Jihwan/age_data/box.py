import os
import json

def adjust_box_dimensions_in_all_folders(base_labeling_dir):
    """
    labeling_data 폴더 안의 모든 하위 폴더에 대해 JSON 파일의 box 정보를 읽고,
    width(w)와 height(h) 중 큰 값을 기준으로 정사각형 박스로 조정합니다.
    """
    for folder_name in os.listdir(base_labeling_dir):
        folder_path = os.path.join(base_labeling_dir, folder_name)
        if os.path.isdir(folder_path):
            print(f"폴더 처리 중: {folder_name}")
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.endswith(".json"):
                        file_path = os.path.join(root, file)
                        with open(file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)

                        # box 크기 조정
                        for annotation in data.get("annotation", []):
                            box = annotation.get("box", {})
                            w = box.get("w", 0)
                            h = box.get("h", 0)

                            if w and h:
                                max_size = max(w, h)
                                box["w"] = max_size
                                box["h"] = max_size
                                print(f"{file}: w와 h를 {max_size}로 조정함.")

                        # 변경된 데이터를 파일에 다시 저장
                        with open(file_path, "w", encoding="utf-8") as f:
                            json.dump(data, f, indent=4, ensure_ascii=False)

    print("\n✅ labeling_data 폴더 내 모든 JSON 파일의 box 크기 조정 완료.")

if __name__ == "__main__":
    BASE_LABELING_DIR = "labeling_data"
    adjust_box_dimensions_in_all_folders(BASE_LABELING_DIR)
