import os
import csv
from tqdm import tqdm
from Module import evaluation

def evaluation_batch(
    dataset_dir, watermarked_dir, qr_origin_dir, qr_extract_dir, log_path="eval_results.csv"
):
    results = []
    image_paths = []
    for root, dirs, files in os.walk(watermarked_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                image_paths.append(os.path.join(root, file))

    for wm_img_path in tqdm(image_paths, desc="Batch Evaluation"):
        rel_path = os.path.relpath(wm_img_path, watermarked_dir)
        dataset_name = os.path.dirname(rel_path)
        img_name = os.path.splitext(os.path.basename(wm_img_path))[0].replace("_wm", "")

        # Path to host image
        host_path = os.path.join(dataset_dir, dataset_name, f"{img_name}.tiff")
        if not os.path.exists(host_path):
            possible_exts = ['.bmp', '.jpg', '.jpeg', '.png']
            found = False
            for ext in possible_exts:
                temp_path = os.path.join(dataset_dir, dataset_name, f"{img_name}{ext}")
                if os.path.exists(temp_path):
                    host_path = temp_path
                    found = True
                    break
            if not found:
                print(f"[WARN] NOT FOUND {rel_path}")
                continue

        # Đường dẫn QR gốc và QR extract
        qr_origin_path = os.path.join(qr_origin_dir, dataset_name, f"{img_name}_qr.png")
        qr_extract_path = os.path.join(qr_extract_dir, dataset_name, f"{img_name}_qr.png")

        # Check QR
        if not (os.path.exists(qr_origin_path) and os.path.exists(qr_extract_path)):
            print(f"[WARN] NO QR: {img_name}")
            continue

        try:
            psnr = evaluation.psnr_cal(host_path, wm_img_path)
            ssim = evaluation.ssim_cal(host_path, wm_img_path)
            nc = evaluation.nc_cal(qr_origin_path, qr_extract_path)
            ber = evaluation.ber_cal(qr_origin_path, qr_extract_path)
            results.append([dataset_name, img_name, psnr, ssim, nc, ber])
        except Exception as e:
            print(f"[ERROR] {img_name}: {e}")
            continue

    # Lưu kết quả ra file CSV
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "img_name", "psnr", "ssim", "nc", "ber"])
        for row in results:
            writer.writerow(row)
    print(f"Saved results to {log_path}")

if __name__ == "__main__":
    dataset_dir      = "Dataset"          
    watermarked_dir  = "Watermarked_image"
    qr_origin_dir    = "Qr_image/Original" 
    qr_extract_dir   = "Qr_image/Qr_extracted" 
    log_path         = "eval_results.csv"

    evaluation_batch(dataset_dir, watermarked_dir, qr_origin_dir, qr_extract_dir, log_path)
