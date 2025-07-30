import qrcode
from PIL import Image
import cv2
import numpy as np

"""
    This module is used to generate and validate the watermark content
    1. generate process containing:
        - receive the data related to the host object (hash value)
        - attach the data to the qr code which have the size is defined by the input and the ecc is Q
        - create the QR image about the data with the expected size and storing it in database
        - scramble the QR code to improve security by arnold tranform
        - return the scramble QR image, which is embedded in the image later
        
    2. validation process containing:
        - taking the extracted QR image from extracting function
        - unscramble the extracted QR image
        - decode it to receive the data
        - (opt) mapping the extracted data with the related information stored in the database 
"""

#define ecc_level
_ECC_LEVEL = {
    "L": qrcode.constants.ERROR_CORRECT_L, #~7% error
    "M": qrcode.constants.ERROR_CORRECT_M, #~15
    "Q": qrcode.constants.ERROR_CORRECT_L, #~25
    "H": qrcode.constants.ERROR_CORRECT_Q  #~30
}

#this function return a Pillow image
def qr_generate(data, output_sz, ecc, version) -> Image.Image:
    #return a PIL image containing a QR code
    #validate ECC
    try:
        ecc_const = _ECC_LEVEL[ecc.upper()]
    except KeyError as e:
        raise ValueError(f"ECC must be one of L, M, Q, H") from e
    
    #create the QRcode object
    qr = qrcode.QRCode(
        version=version, #1-40
        error_correction=ecc_const,
        box_size=3,
        border=4 # or 0
    )
    
    qr.add_data(data)
    #if the version wasn't fixed, pick the smallest version that can hold my data 
    qr.make(fit=True)
    
    #render to PIL image (black and white)
    qr_img = qr.make_image(fill_color="black", back_color="white").convert("L") # L means - grayscale
    
    #resize to requested output_size
    if output_sz is not None:
        qr_img=qr_img.resize((output_sz, output_sz), Image.NEAREST)
        
    return qr_img

# ~~https://github.com/VAD3R-95/QR-code/blob/master/README.md
def qr_decode(filepath):
    img = cv2.imread(filepath)
    detect = cv2.QRCodeDetector()
    data, bbox, _ = detect.detectAndDecode(img)
    return data  

def arnold_trans(img: Image.Image, iter) -> Image.Image:
    #iter - the number of iteration
    #img = qr_img - square image (L/RGB mode)
    
    if img.width != img.height:
        raise ValueError("SQUARE IMAGE!!")
    
    N = img.width
    arr = np.array(img)
    
    #(x',y') = (x+y, x+2y) mod N
    xs, ys = np.meshgrid(np.arange(N), np.arange(N))
    for _ in range(iter):
        xs, ys = (xs+ys) % N, (xs+2*ys)%N
        
    output = arr[ys, xs] #pixle follow new mapping
    # print(output)
    # print(len(output))
    return Image.fromarray(output)  

def inverse_arnold(img: Image.Image, iter) -> Image.Image:
    if img.width != img.height:
        raise ValueError("SQUARE IMAGE!!!")
    
    N = img.width
    arr = np.asarray(img)
    xs, ys = np.meshgrid(np.arange(N), np.arange(N))
    
    for _ in range(iter):
        xs, ys = (2*xs-ys)%N, (-xs+ys)%N
        
    recover = arr[ys, xs]
    # print(len(recover))
    return Image.fromarray(recover) 

def scramble_clm(img, x0, u):
    """
    return Xor image
    """
    h, w = img.shape
    key = clm(x0, u, h, w)
    return np.bitwise_xor(img.astype(np.uint),key)
            
def unscramble_clm(img_scr, x0, u):
    return scramble_clm(img_scr, x0, u) #xor twice with the same key

def gen_watermark(data, output_sz, ecc, version, iter) -> Image.Image:
    qr_img = qr_generate(data, output_sz, ecc, version)
    qr_scramble = arnold_trans(qr_img, iter)
    return qr_scramble #image

# img = qr_generate(data="help me", output_sz=128, ecc="H", version=None)
# print(np.asarray(img).shape)
# img.save("qr_test.png")

# # # img.show()

# img2 = qr_generate(data="help me", output_sz=256, ecc="H", version=None)
# # img.save("qr_256.png")

# print(qr_decode("qr_test.png"))
# print(qr_decode("qr_256.png"))
# arimg=arnold_trans(img2, 10)
# rec = inverse_arnold(arimg, 10)






    