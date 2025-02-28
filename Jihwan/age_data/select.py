import os

def save_subfolder_names(folder_path, output_file):
    """
    주어진 폴더 경로 아래의 모든 하위 폴더명을 리스트 형태로 텍스트 파일에 저장합니다.

    Args:
        folder_path (str): 탐색할 상위 폴더 경로
        output_file (str): 결과를 저장할 텍스트 파일 경로
    """
    # 하위 폴더명 리스트 추출
    subfolders = [name for name in os.listdir(folder_path)
                  if os.path.isdir(os.path.join(folder_path, name))]

    # 리스트를 텍스트 파일로 저장
    with open(output_file, 'w') as f:
        f.write(str(subfolders))  # 리스트 형태로 저장
    print(f"✅ 하위 폴더명이 '{output_file}' 파일에 저장되었습니다.")

if __name__ == "__main__":
    folder_path = "source_data"               # 하위 폴더명을 추출할 상위 폴더 경로
    output_file = "subfolder_names.txt"  # 저장할 텍스트 파일명

    save_subfolder_names(folder_path, output_file)
