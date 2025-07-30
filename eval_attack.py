import os
import csv
import re
from Module import evaluation
from tqdm import tqdm

def get_img_base_from_qr_extract_name(fname):
    match = re.match(r"(.+?)(_wm.*)?_qr$", fname)
    if match:
        return match.group(1)
    if fname.endswith("_qr"):
        return fname[:-3]
    return fname

def eval_qr_nc_ber(qr_origin_dir, qr_extract_dir, log_path="nc_ber_attack_log.csv"):
    rows = []
    for root, dirs, files in os.walk(qr_extract_dir):
        for file in tqdm(files, desc=f"Processing {os.path.basename(root)}"):
            if not file.endswith("_qr.png"):
                continue
            att_path = os.path.join(root, file)
            rel_path = os.path.relpath(att_path, qr_extract_dir)
            dataset_name = os.path.dirname(rel_path)
            fname = os.path.splitext(os.path.basename(att_path))[0]
            img_base = get_img_base_from_qr_extract_name(fname)
            attack_type = "original"
            parts = fname.split('_')
            if 'wm' in parts:
                idx = parts.index('wm')
                if idx + 1 < len(parts) - 1:
                    attack_type = parts[idx + 1]
                elif idx + 1 == len(parts) - 1:
                    attack_type = "none"
            elif len(parts) > 2:
                attack_type = parts[-2]
            qr_origin_path = os.path.join(qr_origin_dir, dataset_name, f"{img_base}_qr.png")
            if not os.path.exists(qr_origin_path):
                print(f"NOT FOUND original QR: {qr_origin_path}")
                continue
            try:
                nc = evaluation.nc_cal(qr_origin_path, att_path)
                ber = evaluation.ber_cal(qr_origin_path, att_path)
            except Exception as e:
                nc, ber = None, None
                print(f"[ERROR] {img_base}-{attack_type}: {e}")
            rows.append([dataset_name, img_base, attack_type, att_path, nc, ber])

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "img_name", "attack", "extract_qr_path", "nc", "ber"])
        for row in rows:
            writer.writerow(row)
    print(f"Saved QR NC/BER log to {log_path}")

if __name__ == "__main__":
    qr_origin_dir = "Qr_image/Original"
    qr_extract_dir = "Qr_image/Qr_extracted"
    log_path = "nc_ber_attack_log.csv"
    eval_qr_nc_ber(qr_origin_dir, qr_extract_dir, log_path)
