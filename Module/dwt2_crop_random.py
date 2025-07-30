import numpy as np
import random
import math
from PIL import Image
import json
from Module import techniques, img_process, generate_qr
import imagehash


def allowed_patch_area(h0, w0, LL2_shape, max_side):
    h, w = LL2_shape
    h_goc = h0 // 4
    w_goc = w0 // 4
    side = min(h_goc, w_goc, max_side)
    return h_goc, w_goc, side

def pick_patch_position_limited(h_goc, w_goc, side, key):
    # key: int or string 
    if isinstance(key, str):
        seed = int.from_bytes(key.encode(), 'little')
    else:
        seed = int(key)
    random.seed(seed)
    x = random.randint(0, h_goc - side)
    y = random.randint(0, w_goc - side)
    return x, y



def embed(
    img_path: str,
    data: str,
    alpha: float,
    meta_save_path: str,
    key,
    qr_save_path: str = None    # folder store QR
):
    # Read original image
    host = Image.open(img_path) #object (L, RGB, YCbCr)
    h0, w0 = host.size[::-1]
    padded = img_process.padding(host, multiple=4, verbose=False)

    # DWT level 2
    LL2, detail_coeffs, cb, cr = techniques.dwt(padded, level=2)
    details2 = detail_coeffs[0]
    details1 = detail_coeffs[1]

    h, w = LL2.shape
    h_goc, w_goc, side = allowed_patch_area(h0, w0, (h, w), 128)
    x, y = pick_patch_position_limited(h_goc, w_goc, side, key)
    LL2_crop = LL2[x:x+side, y:y+side]

    # create original QR without scramble
    qr_img = generate_qr.gen_watermark(data, output_sz=side, ecc="Q", version=None, iter=0)
    if qr_save_path is not None:
        qr_img.save(qr_save_path)

    # scramble QR image
    w_img = generate_qr.gen_watermark(data, output_sz=side, ecc="Q", version=None, iter=10)
    Uw, Sw, Vtw = techniques.svd(np.asarray(w_img, dtype=float))

    # DCT + SVD on embedded area in LL2
    dct_LL2 = techniques.dct2(LL2_crop)
    Uh, Sh, Vth = techniques.svd(dct_LL2)
    Sh_embed = Sh.copy()
    Sh_embed[:side] += alpha * Sw[:side]
    dct_LL2_embed = techniques.recover_svd(Uh, Sh_embed, Vth)
    # Merge embedded area in original LL2 at position (x, y)
    LL2_new = LL2.copy()
    LL2_new[x:x+side, y:y+side] = techniques.idct2(dct_LL2_embed)

    # IDWT 2 level
    rec_y = techniques.idwt2(LL2_new, details2, details1, wavelet="haar")
    rec_y = np.clip(np.rint(rec_y), 0, 255).astype(np.uint8)

    # Convert to RGB
    rec_y_img = Image.fromarray(rec_y)
    cb_img = Image.fromarray(cb)
    cr_img = Image.fromarray(cr)
    merged = Image.merge("YCbCr", (rec_y_img, cb_img, cr_img)).convert("RGB")
    result = img_process.cropping(merged, (h0, w0))
    phash_str = str(imagehash.phash(result))

    # Store metadata 
    metadata = {
        "original_sz": [int(h0), int(w0)],
        "embed_side": int(side),
        "embed_xy": [int(x), int(y)],
        "LL2_shape": [int(h), int(w)],
        "alpha": float(alpha),
        "Uw": Uw.tolist(),
        "Vw": Vtw.tolist(),
        "Sh_pre": Sh[:side].tolist(),
        "phash": phash_str,
        "key": key,
        "message": data     
    }
    with open(meta_save_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    # Return watermarked image, meta and original QR
    return result, metadata, qr_img



def extract(wm_img_path: str, meta_path: str, qr_save_path: str):
    # Load metadata
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    side = meta["embed_side"]
    x, y = meta["embed_xy"]
    alpha = meta["alpha"]
    Uw = np.array(meta["Uw"])
    Vw = np.array(meta["Vw"])
    Sh_pre = np.array(meta["Sh_pre"])
    h0, w0 = meta["original_sz"]

    # Read the watermarked image (input image)
    padded = img_process.padding(wm_img_path, multiple=4, verbose=False)
    LL2, _, _, _ = techniques.dwt(padded, level=2)

    # Redefined the selected LL2 area at position (x, y)
    LL2_crop = LL2[x:x+side, y:y+side]

    # DCT + SVD
    dct_LL2 = techniques.dct2(LL2_crop)
    _, Sh_wm, _ = techniques.svd(dct_LL2)
    real_k = min(len(Sh_wm), len(Sh_pre), Uw.shape[1], Vw.shape[0])
    Sw_hat = (Sh_wm[:real_k] - Sh_pre[:real_k]) / alpha

    Uw_k = Uw[:, :real_k]
    Vw_k = Vw[:real_k, :]
    W_recon = Uw_k @ np.diag(Sw_hat) @ Vw_k
    W_recon = W_recon - W_recon.min()
    if W_recon.max() > 0:
        W_recon = (W_recon / W_recon.max()) * 255
    W_recon = np.clip(np.rint(W_recon), 0, 255).astype(np.uint8)

    binarized = (W_recon > 127).astype(np.uint8) * 255
    W_img = Image.fromarray(binarized)
    unscrambled = generate_qr.inverse_arnold(W_img, 10)
    #Store the extracted QR image
    unscrambled.save(qr_save_path)
    # Return the message decode from extracted QR
    return generate_qr.qr_decode(qr_save_path)
