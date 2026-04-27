# Offroad Terrain Semantic Segmentation
### Team: [YOUR TEAM NAME]
### BigRock Exchange Hackathon — Duality AI Segmentation Track

---

## Project Overview
Semantic segmentation of desert off-road terrain images into 11 classes
using a DINOv2 backbone with a custom ConvNeXt-style segmentation head.

Classes: Background, Trees, Lush Bushes, Dry Grass, Dry Bushes,
Ground Clutter, Flowers, Logs, Rocks, Landscape, Sky

---

## Final Results
| Metric | Baseline | Our Model |
|---|---|---|
| Mean IoU | 0.1968 | 0.3395 |
| Pixel Accuracy | 0.69 | 0.846 |
| Val IoU | 0.2707 | 0.4012 |

---

## Environment Requirements
- OS: Windows 10/11
- Python: 3.10
- CUDA: 11.8
- GPU: NVIDIA RTX 4050 (6GB VRAM)

---

## Installation — Step by Step

### Step 1 — Create Conda Environment
```bash
conda create -n EDU python=3.10 -y
conda activate EDU
```

### Step 2 — Install PyTorch
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 3 — Install Other Libraries
```bash
pip install opencv-python pillow matplotlib tqdm numpy
```

### Step 4 — Verify GPU
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
Expected output: True

---

## Folder Structure
```
Offroad_Segmentation/
├── Offroad_Segmentation_Training_Dataset/
│   ├── train/
│   │   ├── Color_Images/
│   │   └── Segmentation/
│   └── val/
│       ├── Color_Images/
│       └── Segmentation/
├── Offroad_Segmentation_testImages/
│   └── Color_Images/
└── Offroad_Segmentation_Scripts/
    ├── train_segmentation.py
    ├── test_segmentation.py
    ├── visualize.py
    ├── segmentation_head.pth
    ├── segmentation_head_best.pth
    ├── train_stats/
    └── predictions/
```

---

## How to Train the Model

```bash
conda activate EDU
cd Offroad_Segmentation_Scripts
python train_segmentation.py
```

Expected output during training:
```
Epoch 1/30 [Train]: loss=1.234 ...
Epoch 1/30 [Val]: IoU=0.21 ...
✓ New best model saved (IoU: 0.21)
```

Training time: approximately 4-5 hours on RTX 4050
Model is saved to: segmentation_head.pth
Best model saved to: segmentation_head_best.pth

---

## How to Test the Model

```bash
conda activate EDU
cd Offroad_Segmentation_Scripts
python test_segmentation.py
```

Expected output:
```
Processing 1002 test images...
Mean IoU: [score]
Predictions saved to: predictions/
```

---

## How to Visualize Predictions

```bash
python visualize.py
```
Opens a window showing color-coded segmentation overlaid on test images.

---

## Key Improvements Over Baseline

1. DiceCE combined loss with class weights
2. AdamW optimizer with CosineAnnealingLR
3. ASPP multi-scale segmentation head
4. Joint augmentation (flip, color jitter, rotation)
5. 30 epochs with best-model checkpointing
6. Partial DINOv2 backbone fine-tuning (last 2 blocks)
7. Higher input resolution (644x364)

---

## Expected Outputs
- predictions/ folder contains PNG segmentation maps for all 1002 test images
- Each pixel value corresponds to a class ID (0-10)
- train_stats/ contains loss curves and IoU graphs per epoch

---

## Important Notes
- Do NOT use test images for training — strictly separated
- Model trained exclusively on provided Falcon dataset
- Best model checkpoint used for final predictions (not last epoch)

---

## Team Members
- [NAME 1] — AI Engineering
- [NAME 2] — Documentation & Presentation
