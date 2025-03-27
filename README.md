# Virtual Try-On (VTON) Garment Masking Pipeline

This project demonstrates a lightweight and interpretable pipeline for garment masking. Given an image of a person (pose) and a garment image, the system segments the person and generates a binary mask corresponding to the predicted garment class.

## Project Highlights

- Contains experimentations for performing human segmenation with **Self-Correction-Human-Parsing** (LIP and Pascal based) and **Single-Human-Parsing-LIP** models. 
- Uses pretrained **SCHP-LIP** model for fine-grained human segmentation.
- Leverages **CLIP** zero-shot classification to determine garment type.
- Rule-based logic to apply masks for appropriate body regions based on predicted garment type.
- Outputs overlaid images showing the masking step, forming the basis for future garment transfer.

---
## Setup 

This project requires a CUDA-enabled GPU and compatible drivers to run efficiently. Specifically the SCHP models rely on PyTorch with CUDA for accelerated inference. Make sure your environment contains the dependencies listed under ```requirements.txt```. Please download the following models and place them in the models/ folder:
- [PSPNet](https://drive.google.com/drive/folders/1ZOZ9hU3LEADvyR25Orur9mGkkpN8drY5)
- [SHCP (LIP)](https://drive.google.com/file/d/1k4dllHpu0bdx38J7H28rVVLpU-kOHmnH/view)
- [SHCP (Pascal)](https://drive.google.com/file/d/1E5YwNKW2VOEayK9mWCS3Kpsxf-3z04ZE/view)

The end to end pipeline can be run by uncommenting and modifying the final cell in the notebook. The notebook provides a sequential walkthrough of how the pipeline runs. 

---

## Directory Structure

```
VTON/
├── models/                         # Place pretrained models here
├── sample-data/                    # Contains sample input images
│   ├── garments/                   # Garment images used in try-on
│   └── poses/                      # Person images (poses)
│
├── Self-Correction-Human-Parsing/  # SCHP-based segmentation
│   ├── modules/
│   ├── networks/
│   └── utils/
│   └── LICENSE
│
├── Single-Human-Parsing-LIP/       # PSPNet-based segmentation (LIP)
│   ├── net/
│   └── LICENSE
│
├── requirements.txt                # Dependencies file
├── vton.ipynb                      # Main Jupyter notebook
└── README.md                       # Project description & instructions
```

---
