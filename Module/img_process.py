"""
    - Embedding the watermark in the Y channel only 
    This module is used to process the images that do not satisfy the conditions
    Conditions:
        - the size of the image need to be dividible to 16 
        

"""

from PIL import Image
import numpy as np
from typing import Tuple, Union
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

def padding(img, multiple=4, fill=0, verbose=False):
    if isinstance(img, str):
        img = Image.open(img)
    w, h = img.size
    pad_w = (multiple - w % multiple) % multiple
    pad_h = (multiple - h % multiple) % multiple

    # Auto select fill value by image mode
    if img.mode == 'L':
        # Nếu fill là tuple, lấy phần tử đầu
        if isinstance(fill, tuple):
            fill = fill[0]
        fill_correct = int(fill)
    elif img.mode == 'RGB':
        # Nếu fill là int, chuyển thành tuple
        if isinstance(fill, int):
            fill_correct = (fill, fill, fill)
        else:
            fill_correct = fill
    else:
        # Nếu mode khác nữa, dùng 0 hoặc fill mặc định
        fill_correct = fill

    padded = Image.new(img.mode, (w + pad_w, h + pad_h), color=fill_correct)
    padded.paste(img, (0, 0))
    if verbose:
        print(f"[padding] mode={img.mode}, size in: {img.size}, out: {(w+pad_w, h+pad_h)}")
    return padded


def cropping(padded_img: Image.Image, original_sz: Tuple[int, int]) -> Image.Image:
    """
    Crop off the right, bottom of the image
    
    Input:
        padded image
        original size of image
        
    Output:
        cropped image
    """
    h0, w0 = original_sz
    return padded_img.crop((0,0,w0,h0))
    
def split_to_block(
        img: Union[str, Path, Image.Image, np.ndarray],
        block_sz: int = 8
    ) -> np.ndarray:
    """
    split into block
    """
    # img = padding(img)
    # img = Image.open(img)    
    # arr = np.asarray(img)
    # — Load / normalize to HxW array —
    if isinstance(img, (str, Path)):
        arr = np.asarray(Image.open(img).convert("L"), float)
    elif isinstance(img, Image.Image):
        arr = np.asarray(img.convert("L"), float)
    elif isinstance(img, np.ndarray):
        if img.ndim == 3 and img.shape[2] == 1:
            arr = img[...,0].astype(float)
        elif img.ndim == 2:
            arr = img.astype(float)
        else:
            raise ValueError(f"Unsupported array shape for splitting: {img.shape}")
    else:
        raise TypeError(f"Unsupported type: {type(img)}")
    
    h,w = arr.shape
    if h % block_sz or w % block_sz:
        raise ValueError(f"Array shape {h}×{w} not divisible by {block_sz}")
    
    nH, nW = h//block_sz, w//block_sz
    # reshape & swapaxes to get (nH, nW, bs, bs)
    blocks = (arr
              .reshape(nH, block_sz, nW, block_sz)
              .swapaxes(1,2))
    return blocks
    

# padding("qr_test.png").show()
# padding("I01.BMP").show()
# padding("1677_mainfoto_05.jpg").show()

#print(split_to_block("qr_test.png").shape) - ex: (16, 16, 8, 8, 1)

    
    
    