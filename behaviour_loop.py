import os
import cv2
import json
import csv
import numpy as np
import imagehash
from PIL import Image
from tqdm import tqdm
from Module import generate_qr, dwt2_crop_random

# Path
mess_dir = "behaviour/mail/"
metadata_dir = "Metadata/"
output_csv = "qr_extract_mail_results.csv" #change name (mess, zalo)
qr_extract_dir = "qr_extract_platform/"
os.makedirs(qr_extract_dir, exist_ok=True)

#compute pHash to retrie metadata file
def compute_phash(image_path):
    image = Image.open(image_path).convert("L")
    return str(imagehash.phash(image))

def load_all_metadata(metadata_dir):
    phash_to_meta = {}
    for root, _, files in os.walk(metadata_dir):
        for file in files:
            if file.endswith(".json"):
                with open(os.path.join(root, file), "r") as f:
                    meta = json.load(f)
                    phash = meta.get("phash")
                    if phash:
                        phash_to_meta[phash] = (os.path.join(root, file), meta)
    return phash_to_meta

def find_closest_match(phash, phash_to_meta, max_distance=5):
    best_match = None
    best_dist = max_distance + 1
    for saved_phash in phash_to_meta:
        dist = imagehash.hex_to_hash(saved_phash) - imagehash.hex_to_hash(phash)
        if dist < best_dist:
            best_match = phash_to_meta[saved_phash]
            best_dist = dist
    return best_match if best_match else None

# Load metadata
phash_to_meta = load_all_metadata(metadata_dir)

# Process
rows = []

for file in tqdm(os.listdir(mess_dir), desc="Extracting QR from mess"):
    if not file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(mess_dir, file)
    phash = compute_phash(image_path)
    matched = find_closest_match(phash, phash_to_meta)

    if not matched:
        rows.append([file, "", "", "no_match", phash])
        continue

    meta_path, meta = matched
    qr_save_path = os.path.join(qr_extract_dir, f"{file}_qr.png")

    try:
        msg = dwt2_crop_random.extract(image_path, meta_path, qr_save_path)
        status = "extract_success" if msg else "qr_decode_fail"
    except Exception as e:
        msg = ""
        status = f"extract_fail"

    rows.append([file, os.path.basename(meta_path), msg, status, phash])

# === Ghi file CSV ===
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_name", "matched_meta", "decoded_message", "status", "phash"])
    writer.writerows(rows)

print(f"\nResults saved to {output_csv}")
