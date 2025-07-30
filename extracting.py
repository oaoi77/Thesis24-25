import os
import json
from tqdm import tqdm
from Module import dwt2_crop_random

def batch_extract(watermarked_dir, meta_dir, qr_extract_dir):
    # Quét tất cả watermarked image
    image_paths = []
    for root, dirs, files in os.walk(watermarked_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                image_paths.append(os.path.join(root, file))

    for wm_img_path in tqdm(image_paths, desc="Batch Extract"):
        # Xác định relative path và meta path
        rel_path = os.path.relpath(wm_img_path, watermarked_dir)
        dataset_name = os.path.dirname(rel_path)
        img_name = os.path.splitext(os.path.basename(wm_img_path))[0].replace("_wm", "")
        meta_path = os.path.join(meta_dir, dataset_name, f"{img_name}_meta.json")

        # Tạo thư mục output QR extract
        out_qr_folder = os.path.join(qr_extract_dir, dataset_name)
        os.makedirs(out_qr_folder, exist_ok=True)
        qr_extract_path = os.path.join(out_qr_folder, f"{img_name}_qr.png")

        # Trích watermark (decode QR extract)
        try:
            msg = dwt2_crop_random.extract(
                wm_img_path=wm_img_path,
                meta_path=meta_path,
                qr_save_path=qr_extract_path
            )
            print(f"[OK] {rel_path}: {msg}")
        except Exception as e:
            print(f"[ERROR] {rel_path}: {e}")

if __name__ == "__main__":
    watermarked_dir = "Watermarked_image"
    meta_dir = "Metadata"
    qr_extract_dir = "Qr_image/Qr_extracted"

    batch_extract(watermarked_dir, meta_dir, qr_extract_dir)
