import os
import shutil

# 원본 이미지 폴더 경로
image_dir = "/data/ephemeral/home/final_data"

# # 폴더 내 모든 파일 가져오기
# for filename in os.listdir(image_dir):
#     # 파일이 .png인지 확인
#     if filename.endswith(".png") and "_crop_1" in filename:
#         # 새 파일명 생성
#         new_filename = filename.replace("_crop_1", "")
        
#         # 기존 경로와 새로운 경로 설정
#         old_path = os.path.join(image_dir, filename)
#         new_path = os.path.join(image_dir, new_filename)
        
#         # 파일 이름 변경 (덮어쓰기 X, 별도 저장)
#         shutil.move(old_path, new_path)
#         print(f"✅ {filename} → {new_filename} 변경 완료!")

# print("🎉 모든 파일 이름 변경 완료!")

# 이미지 파일 확장자 리스트 (필요하면 추가 가능)
image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp")

# 디렉토리 내 이미지 파일 개수 계산
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(image_extensions)]
num_images = len(image_files)

# 결과 출력
print(f"📂 '{image_dir}' 내 이미지 개수: {num_images}개")