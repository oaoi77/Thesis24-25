import os
import csv
from tqdm import tqdm
from Module import dwt2_crop_random, attack_test

def attack_batch(
    watermarked_dir, meta_dir, attack_output_dir, qr_extract_dir, log_path="attack_eval_log.csv"
):
    # Scan all image from path
    image_paths = []
    for root, dirs, files in os.walk(watermarked_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                image_paths.append(os.path.join(root, file))

    log_rows = []
    for wm_img_path in tqdm(image_paths, desc="Batch Attack"):
        rel_path = os.path.relpath(wm_img_path, watermarked_dir)
        dataset_name = os.path.dirname(rel_path)
        img_name = os.path.splitext(os.path.basename(wm_img_path))[0].replace("_wm", "")
        meta_path = os.path.join(meta_dir, dataset_name, f"{img_name}_meta.json")
        # Create folder to store attacked image
        attack_out_folder = os.path.join(attack_output_dir, dataset_name)
        os.makedirs(attack_out_folder, exist_ok=True)
        # Perform all attacks 
        attacked_paths = attack_test.perform_all_attacks_on_watermarked_image(
            wm_img_path, out_dir=attack_out_folder
        )
        # Extract QR on each attacked image
        for att_img_path in attacked_paths:
            attack_type = os.path.basename(att_img_path).replace(f"{img_name}_", "").split('.')[0]
            # Store extracted QR of each
            qr_out_folder = os.path.join(qr_extract_dir, dataset_name)
            os.makedirs(qr_out_folder, exist_ok=True)
            qr_save_path = os.path.join(qr_out_folder, f"{img_name}_{attack_type}_qr.png")
            try:
                qr_msg = dwt2_crop_random.extract(
                    wm_img_path=att_img_path,
                    meta_path=meta_path,
                    qr_save_path=qr_save_path
                )
            except Exception as e:
                qr_msg = f"ERROR: {e}"
            log_rows.append([dataset_name, img_name, attack_type, att_img_path, qr_save_path, qr_msg])

    # Log
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "img_name", "attack", "attacked_img_path", "qr_img_path", "qr_msg"])
        for row in log_rows:
            writer.writerow(row)
    print(f"Saved attack+extract log to {log_path}")

if __name__ == "__main__":
    watermarked_dir   = "Watermarked_image"
    meta_dir          = "Metadata"
    attack_output_dir = "Attack_output"
    qr_extract_dir    = "Qr_image/Qr_extracted"
    log_path          = "attack_eval_log.csv"

    attack_batch(watermarked_dir, meta_dir, attack_output_dir, qr_extract_dir, log_path)
