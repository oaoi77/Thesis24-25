import os
import cv2
import numpy as np
from PIL import Image
from skimage.util import random_noise

def perform_all_attacks_on_watermarked_image(im_wm_path, out_dir=None, attacks=None):
    """
    Perform attack on watermarked image, store attacked watermarked in folder
    Args:
        im_wm_path: watermarked image path.
        out_dir: Folder storing attack image.
        attacks: choose attack to run (default is all).
    Returns:
        Attacked image path.
    """
    if out_dir is None:
        out_dir = os.path.dirname(im_wm_path)
    os.makedirs(out_dir, exist_ok=True)
    img_name = os.path.splitext(os.path.basename(im_wm_path))[0]

    # Load image (PIL -> np)
    img = Image.open(im_wm_path).convert("RGB")
    img_np = np.array(img)
    h, w, _ = img_np.shape

    attacked_paths = []

    # attack list 
    all_attacks = {
        "blur": lambda img: cv2.GaussianBlur(img, (5,5), 0),
        "gaussian_noise": lambda img: np.clip(img + np.random.normal(0, 15, img.shape), 0, 255).astype(np.uint8),
        "salt_pepper_noise": lambda img: (255*random_noise(img, mode='s&p', amount=0.01)).astype(np.uint8),
        "jpeg": None, 
        "crop": lambda img: img[int(0.2*h):h-int(0.2*h), int(0.2*w):w-int(0.2*w)],
        "resize": lambda img: cv2.resize(img, (w//2, h//2), interpolation=cv2.INTER_LINEAR),
        "resize_restore": lambda img: cv2.resize(cv2.resize(img, (w//2, h//2)), (w, h)),
        "sharp": lambda img: cv2.filter2D(img, -1, np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])),
        "hist_eq": lambda img: cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)),
        "gamma": lambda img: cv2.LUT(img, np.array([((i/255.0)**(1/1.5))*255 for i in range(256)]).astype("uint8")),
        "rotation": lambda img: cv2.warpAffine(img, cv2.getRotationMatrix2D((w/2,h/2), 5, 1), (w,h), borderMode=cv2.BORDER_REFLECT),
        "rowcol_remove": lambda img: _row_col_remove(img, row_step=10, col_step=10),
        "local_bend": lambda img: _local_bending(img, amplitude=10, frequency=40),
        "gray": lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2GRAY),
    }

    # attack is None -> run all (default)
    if attacks is None:
        attacks = list(all_attacks.keys())

    for name in attacks:
        if name == "jpeg":
            jpeg_path = os.path.join(out_dir, f"{img_name}_jpeg.jpg")
            img.save(jpeg_path, quality=50)
            attacked_paths.append(jpeg_path)
        else:
            attack_func = all_attacks[name]
            attacked = attack_func(img_np)
            ext = "png" if attacked.ndim != 2 else "png" # dùng png luôn cho mọi loại
            attack_path = os.path.join(out_dir, f"{img_name}_{name}.{ext}")
            Image.fromarray(attacked).save(attack_path)
            attacked_paths.append(attack_path)

    return attacked_paths

# ===== Hàm hỗ trợ attack đặc biệt =====

def _row_col_remove(img, row_step=10, col_step=10):
    out = img.copy()
    out[::row_step,:,:] = 0
    out[:,::col_step,:] = 0
    return out

def _local_bending(img, amplitude=5, frequency=20):
    h, w = img.shape[:2]
    map_y, map_x = np.indices((h, w), dtype=np.float32)
    map_x += amplitude * np.sin(2 * np.pi * map_y / frequency)
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


# attacked = perform_all_attacks_on_watermarked_image("wm_image.png", "attack_results")

