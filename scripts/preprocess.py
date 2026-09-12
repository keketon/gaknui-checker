import os
from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageOps
from targets import TARGETS

def remove_cross_category_duplicates(raw_dir: str = "data/raw") -> None:
    """複数カテゴリに同じ画像（ファイル名=ハッシュが一致）が存在する場合、
    広告等の無関係な画像とみなして全カテゴリから削除する
    """
    raw_path = Path(raw_dir)

    file_locations: dict[str, list[Path]] = defaultdict(list)
    for category_dir in raw_path.iterdir():
        if not category_dir.is_dir():
            continue
        for file_path in category_dir.glob("*"):
            file_locations[file_path.name].append(file_path)

    for file_name, paths in file_locations.items():
        categories = {path.parent.name for path in paths}
        if len(categories) < 2:
            continue

        print(f"Removing cross-category duplicate: {file_name} ({', '.join(categories)})")
        for path in paths:
            path.unlink()

def preprocess_images(input_dir: str, output_dir: str, size: int = 224):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing: {input_path} -> {output_path}")
    
    for file_path in input_path.glob("*"):
      # 収集スクリプトでJPGを保存するようにしているため、JPG以外見つかったらエラーで止める
      if file_path.suffix.lower() not in [".jpg", ".jpeg"]:
          print(f"Skipping non-image file: {file_path}")
          raise ValueError(f"Unsupported file format found: {file_path.suffix}")
      
      try:
          with Image.open(file_path) as img:
              img = img.convert("RGB")
              final_img = ImageOps.pad(img, (size, size), color=(0, 0, 0)) # 黒でパディング
              save_path = output_path / f"processed_{file_path.stem}.jpg"
              final_img.save(save_path, "JPEG", quality=90)
            
      except Exception as e:
          print(f"Error processing {file_path}: {e}") # 破損ファイル/画像じゃないファイルが大体の原因
          
if __name__ == "__main__":
    remove_cross_category_duplicates()

    for target in TARGETS:
        raw_img_dir = f"data/raw/{target['category_name']}"
        destination_dir = f"data/processed/{target['category_name']}"
        preprocess_images(raw_img_dir, destination_dir)
