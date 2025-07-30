import numpy as np
import pywt
from PIL import Image
from typing import Literal, Tuple, Union, List
from scipy.fftpack import dct, idct
from pathlib import Path
import hashlib



# embed the W in Y channel
def y_channel(img_input: Union[str, Path, Image.Image, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:    
    """
    Input: original color image (various format)
    Output: return numpy array (HxW) of Y channel
    
    Step:
    1. Convert to YCbCr -> [:,:,0] --> L
    """
    # Load or accept array
    if isinstance(img_input, Image.Image):
        img = img_input
    elif isinstance(img_input, (str, Path)):
        img = Image.open(img_input)
    elif isinstance(img_input, np.ndarray):
        # assume already channel-stacked or 2D
        arr = img_input
        if arr.ndim == 2:
            # treat as Y, make dummy Cb/Cr
            return arr.astype(np.uint8), arr.astype(np.uint8), arr.astype(np.uint8)
        # if H×W×C, convert via PIL
        img = Image.fromarray(arr)
    else:
        raise TypeError(f"Unsupported type: {type(img_input)}")

    # Ensure YCbCr
    if img.mode != "YCbCr":
        ycbcr = img.convert("YCbCr")
    else:
        ycbcr = img

    y, cb, cr = ycbcr.split()
    return (np.asarray(y, dtype=np.uint8),
            np.asarray(cb, dtype=np.uint8),
            np.asarray(cr, dtype=np.uint8))

def dwt(
    img,
    wavelet: str ="haar",
    level: int = 1,
    mode: Literal["periodization", "symmetric", "zero"] = "periodization"
) -> Tuple[np.ndarray, list]:
    y, cb, cr  = y_channel(img)
    img_y = y.astype(np.float32)
    #print(img_y.shape)
    coeffs = pywt.wavedec2(img_y, wavelet=wavelet, level=level, mode=mode)
    #LL_bands = coeffs[0].shape (cA = LL)
    cA, *detail_coeffs = coeffs
    return cA, detail_coeffs, cb, cr

def idwt(
    cA: np.ndarray, 
    detail_coeffs: list,
    wavelet: str = "haar",
    mode: Literal["periodization", "symmetric", "zero"] = "periodization"
) -> Image.Image:
    """inverse to get back the image """
    coeffs = [cA, *detail_coeffs]
    rec = pywt.waverec2(coeffs, wavelet=wavelet, mode=mode)
    rec_uint8 = np.clip(np.rint(rec), 0, 255).astype(np.uint8)
    return Image.fromarray(rec_uint8, mode="L")

def idwt2(
    LL2: np.ndarray,
    details2: tuple,
    details1: tuple,
    wavelet: str = "haar",
    mode: str = "periodization"
) -> np.ndarray:
    LL1 = pywt.waverec2([LL2, details2], wavelet=wavelet, mode=mode)
    rec_y = pywt.waverec2([LL1, details1], wavelet=wavelet, mode=mode)
    return rec_y

    
def dct2(blocks: np.ndarray) -> np.ndarray:
    """
    input: np.ndarray 2D (bh x bw) - in this thesis: 8x8
        type of data: float32 or float64 (if uint8 -> float)
    output: numpy.ndarray + shape; AC and DC coeff; matrix DCT 
    """
    return dct(dct(blocks, axis=0, norm='ortho'), axis=1, norm='ortho')

def idct2(blocks):
    return idct(idct(blocks, axis=0, norm='ortho'), axis=1, norm="ortho")

#avoid to select the blocks containing padding part 
# def valid_block(original_sz: Tuple[int, int],
#                 multiple: int = 16,
#                 dwt_level: int=1,
#                 block_size: int=8) -> np.ndarray:
#     """
#     - compute a boolean mask of shape (hxw) in the LL band contain no padding px
#     Input:
#     - original_sz: the original size of the image
#     - multiple: padding multiple for host
#     - dwt_level: level of DWT
#     - block_size: DCT block size
    
#     Output:
#     mask: ndarray bool 
#         T - block is fully inside original data
#         F - block in pad zone
#     """
#     h0, w0 = original_sz
    
#     # how much was padded on host
#     pad_h = (-h0) % multiple
#     pad_w = (-w0) % multiple
    
#     # dimension of LL after dwt_level
#     h_LL = (h0+pad_h)//(2**dwt_level)
#     w_LL = (w0+pad_w)//(2**dwt_level)
    
#     # padding part in LL 
#     pad_h_LL = pad_h // (2**dwt_level)
#     pad_w_LL = pad_w // (2**dwt_level)
    
#     # total block
#     nh = h_LL // block_size
#     nw = w_LL // block_size
    
#     mask = np.ones((nh, nw), dtype=bool)
    
#     if pad_w_LL:
#         bad_cols = pad_w_LL//block_size
#         mask[:, -bad_cols:] =False
        
#     if pad_h_LL:
#         bad_rows = pad_h_LL //block_size
#         mask[-bad_rows:, :] = False
#     return mask

# def select_blks(mask: np.ndarray,
#                 m: int,
#                 key: Union[str, int]
#                 ) -> List[Tuple[int, int]]:
#     nh, nw = mask.shape
#     coords = [(i,j) for i in range(nh) for j in range(nw) if mask[i,j]]
#     if m > len(coords):
#         raise ValueError(f"Need {m} blocks but only {len(coords)} valid")
#     #derive 64-bit seed from key
#     if isinstance(key, str):
#         seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
#     else:
#         seed = key & ((1<<64)-1)
    
#     rng = np.random.default_rng(seed)
#     pick = rng.choice(len(coords), size=m, replace=False)
#     return [coords[k] for k in pick]
        
# def block_selection(h: int, w: int, m: int, key: Union[int,str]) -> List[Tuple[int, int]]:
#     """
#     pseudo-random select m distinct blocks out of an HxW grid, using PRNG by 'key'
#     return list of (row, col) indices
#     """
#     if isinstance(key, str):
#         digest = hashlib.sha256(key.encode("utf-8")).digest()
#         seed = int.from_bytes(digest[:8], "big")
#     else:
#         seed=int(key) & ((1<<64)-1)
        
#     #init a reproducible PRNG
#     rng = np.random.default_rng(seed)
#     total = h*w
#     if m>total:
#         raise ValueError(f"Cannot select {m} from {total} total blocks")
    
#     # m unique linear indeices, then convert to 2D 
#     flat_idx = rng.choice(total, size=m, replace=False)
#     return [(idx//w, idx%w) for idx in flat_idx]


#---SVD---
def svd(matrix: np.ndarray, full: bool=False):
    """
    Decompose: A = U*Vt*zigma
    
    return:
    U, S, Vt - np.ndarray
    """
    if matrix.ndim != 2:
        raise ValueError("2D matrix required")
    U, S, Vt = np.linalg.svd(matrix.astype(float), full_matrices=full)
    return U,S,Vt
def recover_svd(U: np.ndarray, S: np.ndarray, Vt: np.ndarray) -> np.ndarray:
    # Convert S to diagonal matrix if needed
    if S.ndim == 1:
        S = np.diag(S)
    elif S.ndim != 2:
        raise ValueError("S must be 1D or 2D array")

    if U.shape[1] != S.shape[0] or S.shape[1] != Vt.shape[0]:
        raise ValueError(f"Incompatible shapes: U{U.shape}, S{S.shape}, Vt{Vt.shape}")

    return U @ S @ Vt



#decomposition Y channel image
# cA, details = dwt("I01.BMP", wavelet="haar", level=1)
# cH, cV, cD = details[0]

# print("LL:", cA.shape, "LH:", cH.shape, "HL:", cV.shape, "HH:", cD.shape)

# save_band(cA, "LL.png")

# # 3) Nghịch DWT để xem kết quả
# restored_Y = idwt(cA, details, wavelet="haar")
# restored_Y.save("check.png")
