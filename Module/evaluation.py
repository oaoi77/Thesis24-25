import numpy as np 
import cv2
from skimage.metrics import peak_signal_noise_ratio,  structural_similarity 
import math

#Containing the funnction to evaluate the quality of the image as well as the quality of the watermark
#Including: PSNR, SSIM, NC, BER

def load_image_color(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot load the image: {path}")
    return img

def load_qr(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load the image: {path}")
    return img

#PSNR
def psnr_cal(path1, path2):
    img1 = load_image_color(path1)
    img2 = load_image_color(path2)
    
    if img1.shape != img2.shape:
        raise ValueError(f"Images must have the same size")
    
    mse = np.mean((img1-img2)**2)
    if mse==0:
        return float('inf')
    return 10*math.log10(255.0**2/mse)

#SSIM 
def ssim_cal(path1, path2):
    img1 = load_image_color(path1)
    img2 = load_image_color(path2)
    
    if img1.shape != img2.shape:
        raise ValueError(f"Images must have the same size")
    
    # Convert from BGR (OpenCV) to RGB
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    # Compute SSIM per channel and average
    ssim_total = 0
    for i in range(3):  # RGB channels
        ssim_c, _ = structural_similarity(img1[:, :, i], img2[:, :, i], full=True)
        ssim_total += ssim_c
    return ssim_total / 3

#BER
def ber_cal(path1, path2):
    img1 = load_qr(path1)
    img2 = load_qr(path2)

    if img1.shape != img2.shape:
        raise ValueError("Watermark images must be the same size for BER.")

    img1_bin = (img1 > 127).astype(np.uint8)
    img2_bin = (img2 > 127).astype(np.uint8)

    total_bits = img1_bin.size
    bit_errors = np.sum(img1_bin != img2_bin)
    return bit_errors / total_bits

#NC
def nc_cal(path1, path2):
    img1 = load_qr(path1).astype(np.float32)
    img2 = load_qr(path2).astype(np.float32)

    if img1.shape != img2.shape:
        raise ValueError("Images must be the same shape for NC.")

    numerator = np.sum(img1 * img2)
    denominator = np.sqrt(np.sum(img1 ** 2) * np.sum(img2 ** 2))

    return numerator / denominator if denominator != 0 else 0


# # # Host image vs Watermarked image (color)
# psnr_val = psnr_cal('I01.BMP', 'watermarked_output.png')
# ssim_val = ssim_cal('I01.BMP', 'watermarked_output.png')

# # # QR watermark (grayscale binary)
# # ber_val = ber_cal('images/original_qr.png', 'images/extracted_qr.png')
# # nc_val = nc_cal('images/original_qr.png', 'images/extracted_qr.png')

# print(f"PSNR (color): {psnr_val:.2f} dB")
# print(f"SSIM (color): {ssim_val:.4f}")
# # print(f"BER (QR): {ber_val:.4f}")
# # print(f"NC (QR): {nc_val:.4f}")


    