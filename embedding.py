import hashlib
from Module import dwt2_crop_random
import os
from tqdm import tqdm
import json

def get_img_key(master_seed, img_name, dataset_name=""):
    s = f"{master_seed}_{dataset_name}_{img_name}"
    h = hashlib.sha256(s.encode()).hexdigest()
    key = int(h[:8], 16)
    return key

def batch_embed(input_image_dir, watermarked_dir, meta_dir, alpha, master_seed, raw_message):
    msg_hash = hashlib.sha256(raw_message.encode()).hexdigest()
    print(f"[INFO] QR content for all images: {msg_hash}")

    image_paths = []
    for root, dirs, files in os.walk(input_image_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_paths.append(os.path.join(root, file))

    for img_path in tqdm(image_paths, desc="Batch Embedding"):
        rel_path = os.path.relpath(img_path, input_image_dir)
        dataset_name = os.path.dirname(rel_path)  
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        out_img_folder = os.path.join(watermarked_dir, dataset_name)
        out_meta_folder = os.path.join(meta_dir, dataset_name)
        out_qr_folder = os.path.join("Qr_image/Original", dataset_name)
        os.makedirs(out_img_folder, exist_ok=True)
        os.makedirs(out_meta_folder, exist_ok=True)
        os.makedirs(out_qr_folder, exist_ok=True)

        key = get_img_key(master_seed, img_name, dataset_name)

        try:
            wm_img, meta, qr_img = dwt2_crop_random.embed(
                img_path=img_path,
                data=msg_hash,
                alpha=alpha,
                meta_save_path=os.path.join(out_meta_folder, f"{img_name}_meta.json"),
                key=key,
                qr_save_path=os.path.join(out_qr_folder, f"{img_name}_qr.png")
            )
            wm_img.save(os.path.join(out_img_folder, f"{img_name}_wm.png"))

            
            meta['message_raw'] = raw_message
            meta['message_hash'] = msg_hash
            with open(os.path.join(out_meta_folder, f"{img_name}_meta.json"), "w") as f:
                json.dump(meta, f, indent=4)

            print(f"[OK] {rel_path} | key: {key}")
        except Exception as e:
            print(f"[ERROR] {rel_path} | {e}")

if __name__ == "__main__":
    input_image_dir = "Dataset"              #folder gốc chứa  dataset con
    watermarked_dir = "Watermarked_image"
    meta_dir = "Metadata"
    alpha = 0.06
    master_seed = "my_secret_password"
    raw_message = "myimage_imauthor_yourecipient"

    batch_embed(input_image_dir, watermarked_dir, meta_dir, alpha, master_seed, raw_message)
