import os
from pathlib import Path
from PIL import Image, ImageOps

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
    preprocess_images("data/raw/ことね ぬい", "data/processed/ことね ぬい")
