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

## Datasets
This project test on color images from different datasets, including:
DIV2K dataset: 
https://data.vision.ee.ethz.ch/cvl/DIV2K/ 

Art Dataset for Interactive System & Sensor Data (drawings and paintings):
https://www.kaggle.com/datasets/ziya07/art-dataset-for-interactive-system-and-sensor-data?resource=download

The USC-SIPI Image Database (Miscellaneous): 
https://sipi.usc.edu/database/database.php?volume=misc

tid2013:
https://www.ponomarenko.info/tid2013.htm

Watermarked / Not watermarked images:
https://www.kaggle.com/datasets/felicepollano/watermarked-not-watermarked-images 

> **Note**: However, this project does not install all the images from these datasets, only install a part of them, in details:
> https://drive.google.com/drive/folders/1zw9CTnPA5-lf4QYPp1KOUI8vNhKgqRMP?usp=sharing 

## Folder structure
