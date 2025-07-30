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
Because of a huge number of result images, folders will only contains several sample input/output images. The images will
```
./
├── Module/             # Functions: embed/extract, attack, evaluate
├── Dataset/            # Original image from different datasets
│ ├── DIV2K dataset/
│ ├── tid2013/
│ └── ...
├── Qr_image/           # original and extracted QR image
│ ├── Original/
│ │ ├── DIV2K dataset/
| | └── ...
│ └── Qr_extracted/
│ │ ├── DIV2K dataset/
| | └── ...
├── Metadata/           # JSON metadata files
│ ├── DIV2K dataset/
│ └── ...
├── Watermarked_image/  # Output watermarked images
│ ├── DIV2K dataset/
│ └── ...
├── Attack_output/      # Attacked watermarked images
│ ├── DIV2K dataset/
│ └── ...
├── behaviour/          # Images after transmission simulation
│ ├── mail/
│ ├── zalo/
│ └── mess/
├── requirements.txt    # Python dependencies
├── attack_batch.py     # executing attack list on images
├── behaviour.py        # calculating pHash, retrieving Metadata and extracting QR image on transmitted images
├── embedding.py        # embedding process 
├── extracting.py       # extracting process 
├── eval_attack.py      # evaluation with attack
├── eval_noattack.py    # evaluation without attack 
└── README.md           # Project documentation
```
## Note:
Because of focusing on evaluating the robustness and imperceptibility of watermarking techniques, the Metadata file will be retrieved based on Image file name. For example:

| File Type                   | Example Filename            |
|-----------------------------|-----------------------------|
| Original Image              | `0552.png`                  |
| QR Code Image               | `0552_qr.png`               |
| Watermarked Image           | `0552_wm.png`               |
| Metadata File               | `0552_meta.json`            |
| Attacked Image              | `0552_wm_<attack>.png`      |
| QR extracted                | `0552_wm_<attack>_qr.png`   |
| Upload/Download across Zalo | `1_zalo.png`                |

This allows to evaluate the robustness and imperceptibility of watermarking techniques under the assumption that all input watermarked images can be matched and retrieved the corresponding Metadata file.

On the other hand, the pHash in this thesis just is used for human behaviour simulations case. Suppose the input image is renamed due to uploading and downloading across platforms. In this case, phash will be used to retrieve the Metadata file by finding the hash value closest to the calculated value.