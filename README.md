# Digital Watermarking for Copyright Protection and Tracing Unauthorized Distribution

This project selects and implements a robust digital image watermarking system for copyright protection and traceability. It embeds a scrambled **QR code** watermark into color images using a hybrid approach combining **Discrete Wavelet Transform (DWT)**, **Discrete Cosine Transform (DCT)**, and **Singular Value Decomposition (SVD)**.

The system is tested on 962 images from datasets like DIV2K, TID2013, USC-SIPI, and others (in PNG, JPEG, BMP, TIFF formats) with varied resolutions and aspect ratios. It supports practical scenarios where watermarked images are shared over social platforms (Zalo, Messenger, Email).

The method achieves high imperceptibility in the absence of attacks (PSNR > 40 dB, SSIM > 0.99). Under attacks like JPEG compression, noise, grayscale conversion, and filtering, the watermark remains robust (NC ≈ 1.00, BER between 0.00–0.10). The embedded QR code can still be accurately extracted and decoded after transmission or distortion.

> Techniques used: DWT, DCT, SVD, QR code  
> Application: Copyright protection, forensic watermarking

## Installation

```
#Set up Python environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install numpy Pillow opencv-python pywavelets scipy scikit-image qrcode imagehash

#or 
pip install -r requirements.txt
```

## Folder structure
