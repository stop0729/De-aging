import os
import shutil
import numpy as np

def consolidate_images_and_create_npy(source_data_dir, final_data_dir, npy_filename):
    """
    source_data 폴더 안의 모든 하위 폴더에 있는 'cropped' 폴더 내 PNG 파일을
    final_data 폴더로 이동시키고, 각 파일명과 해당 나이(age)를 매핑하여 .npy 파일로 저장합니다.

    파일명 형식 예시: '0013_1972_06_00000001_F_crop_1.png'
    -> 나이(age_past) 필드는 파일명에서 세 번째 부분(예: '06')을 사용
    """
    os.makedirs(final_data_dir, exist_ok=True)
    
    data_records = []

    for folder_name in os.listdir(source_data_dir):
        folder_path = os.path.join(source_data_dir, folder_name)
        cropped_folder_path = os.path.join(folder_path, "cropped")

        if os.path.isdir(cropped_folder_path):
            for file in os.listdir(cropped_folder_path):
                if file.endswith(".png"):
                    # 파일명에서 나이(age_past) 추출
                    try:
                        age_past = file.split("_")[2]  # 세 번째 부분이 나이 정보
                    except IndexError:
                        age_past = "0"  # 기본값 처리

                    src_path = os.path.join(cropped_folder_path, file)
                    dst_path = os.path.join(final_data_dir, file)

                    shutil.copy2(src_path, dst_path)
                    data_records.append([file, age_past])
                    print(f"이동됨: {file} (나이: {age_past})")

    # .npy 파일 저장
    np.save(os.path.join(final_data_dir, npy_filename), np.array(data_records, dtype='<U21'))
    print(f"\n✅ 모든 이미지가 '{final_data_dir}'로 이동되었으며 '{npy_filename}'가 생성되었습니다.")

if __name__ == "__main__":
    SOURCE_DATA_DIR = "source_data"
    FINAL_DATA_DIR = "final_data"
    NPY_FILENAME = "training_ages.npy"

    consolidate_images_and_create_npy(SOURCE_DATA_DIR, FINAL_DATA_DIR, NPY_FILENAME)
