# Appearance Meets Geometry: A CNN Approach to Segment Ancient Fortification Facades

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)

##  Publication Information

**Conference:** International Conference on Computer Applications in Archaeology (CAA) 2025, Athens

**Paper Title:** Appearance Meets Geometry: Deep Learning for Segmentation and Classification in Archaeological Fortification Studies

**DOI:** 10.5281/zenodo.17011862

**Authors:** Nils Schnorr¹, Thomas Leimkühler²
**Affiliations:** 
- ¹ Saarland University, Institute for Classical Archaeology
- ² Max Planck Institute for Informaticts

**Abstract:** [TO BE FILLED]

## Overview

This repository contains the complete pipeline for semantic segmentation of ancient fortification facades using a multi-channel U-Net architecture. The approach combines geometric information (normal maps derived from heightmaps) with appearance data (orthomosaics) to classify masonry types in archaeological documentation.

### Key Features
- **Multi-channel Input Support:** 3-channel (normal maps only), 4-channel (RGB + alpha), or 7-channel (RGB + normal maps + alpha) configurations
- **Four Masonry Classes:** Background, Ashlar, Polygonal, and Quarry Stone
- **Sobol Sequence Sampling:** Efficient snippet generation for training data
- **Comprehensive Evaluation:** IoU and F1 metrics with confusion matrix visualization

## Repository Structure

```
AppearanceMeetsGeometry/
│
├── 01_image_preparation_for_ML/
│   └── image_preparation_pipeline.ipynb    # Complete data preparation pipeline
│
├── 02_MachineLearning/
│   ├── 3_channel_4_class_PytorchUNET.ipynb # Normal maps only training
│   ├── 4_channel_4_class_PytorchUNET.ipynb # RGB + depth training
│   ├── 7_channel_4_class_PytorchUNET.ipynb # Full multi-channel training
│   └── *.pth                                # Pre-trained models
│
├── 03_image_segmentation_with_trained_ML/
│   ├── 3-channel-segmenting.ipynb          # Inference with 3-channel model
│   ├── 4-channel-segmenting.ipynb          # Inference with 4-channel model
│   └── 7-channel-segmenting.ipynb          # Inference with 7-channel model
│
└── 04_segmentation_evaluation/
    └── segmentation_evaluation.ipynb        # Comprehensive evaluation metrics
```

##  Quick Start

### Prerequisites

```bash
# Create a new conda environment
conda create -n appearance-geometry python=3.9
conda activate appearance-geometry

# Install required packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy opencv-python pillow matplotlib seaborn
pip install rasterio scikit-learn scikit-image
pip install pycocotools sobol-seq tqdm pandas
pip install jupyter notebook ipykernel
```



## Dataset Preparation

### Required Input Data
1. **Orthomosaic Images** (PNG format)
2. **Heightmaps** (GeoTIFF format)
3. **COCO JSON Annotations** (polygon annotations for masonry classes)

### Data Directory Structure
```
your_data_directory/
├── images/              # Original orthomosaics
├── heightmaps/          # Elevation data (GeoTIFF)
├── masks/               # Generated from COCO JSON
├── normalmaps/          # Generated from heightmaps
├── snippets_orthomosaics/
├── snippets_masks/
├── snippets_normalmaps/
└── annotations.json     # COCO format annotations
```

## Complete Pipeline Walkthrough

### Step 1: Image Preparation

1. **Open the preparation notebook:**
   ```bash
   jupyter notebook 01_image_preparation_for_ML/image_preparation_pipeline.ipynb
   ```

2. **Configure paths in the first cell:**
   ```python
   BASE_DIR = "path/to/your/data"
   COCO_JSON_PATH = "path/to/annotations.json"
   CROP_SIZE = 1280  # Snippet size
   DESIRED_COVERAGE = 1.6  # Sobol coverage factor
   ```

3. **Run all cells sequentially** to:
   - Convert COCO annotations to color-coded masks
   - Apply data augmentation (horizontal flipping)
   - Generate normal maps from heightmaps
   - Create training snippets using Sobol sampling
   - Remove invalid (black) masks

**Expected Output:** Directories containing aligned snippets ready for training

### Step 2: Model Training

1. **Choose your input configuration:**
   - `3_channel`: Normal maps only (geometry-focused)
   - `4_channel`: RGB + depth information
   - `7_channel`: RGB + normal maps + depth (full multi-modal)

2. **Open the corresponding training notebook:**
   ```bash
   jupyter notebook 02_MachineLearning/[3/4/7]_channel_4_class_PytorchUNET.ipynb
   ```

3. **Configure training parameters:**
   ```python
   normalsDirectory = "path/to/snippets_normalmaps"
   maskDirectory = "path/to/snippets_masks"
   numEpochs = 300
   modelTrainedName = "your_model_name"
   ```

4. **Run the training pipeline**
   - The notebook will automatically:
     - Load and preprocess data
     - Split into train/validation sets (90/10)
     - Train the U-Net model with Adam optimizer
     - Save checkpoints and visualizations
     - Generate training curves

**Training Time:** ~2-4 hours on RTX 3080 (300 epochs)

### Step 3: Segmentation Inference

1. **Prepare your test images** in the appropriate format

2. **Open the segmentation notebook:**
   ```bash
   jupyter notebook 03_image_segmentation_with_trained_ML/[3/4/7]-channel-segmenting.ipynb
   ```

3. **Load your trained model:**
   ```python
   model_path = "02_MachineLearning/your_model.pth"
   checkpoint = torch.load(model_path)
   model.load_state_dict(checkpoint['model_state_dict'])
   ```

4. **Run inference on new images**
   - The model will output segmentation masks with class predictions

### Step 4: Evaluation

1. **Open the evaluation notebook:**
   ```bash
   jupyter notebook 04_segmentation_evaluation/segmentation_evaluation.ipynb
   ```

2. **Set paths to ground truth and predictions:**
   ```python
   GT_MASK_PATH = "path/to/ground_truth_mask.png"
   PRED_MASK_PATH = "path/to/predicted_mask.png"
   ```

3. **Run evaluation to get:**
   - Per-class IoU scores
   - Precision, Recall, F1-scores
   - Confusion matrix
   - Visual comparisons

## Pre-trained Models

We provide three pre-trained models in the `02_MachineLearning` folder:

| Model | Input Channels | Description | mIoU |
|-------|---------------|-------------|------|
| `2025-08-11_3-channel_4-class-EX_300.pth` | 3 | Normal maps only | [TO BE FILLED] |
| `2025-08-11_4-channel_4-class-EX_300.pth` | 4 | RGB + depth | [TO BE FILLED] |
| `2025-08-11_7-channel_4-class-EX_300.pth` | 7 | Full multi-modal | [TO BE FILLED] |

## Class Color Mapping

| Class | RGB Color | Hex Code |
|-------|-----------|----------|
| Background | Black (0,0,0) | #000000 |
| Ashlar | Blue (0,0,255) | #0000FF |
| Polygonal | Red (255,0,0) | #FF0000 |
| Quarry Stone | Yellow (255,255,0) | #FFFF00 |

## Results

[TO BE FILLED - Add your key results, sample segmentations, and performance metrics]

### Sample Segmentation Results
[Add figure showing input image, ground truth, and prediction]

### Quantitative Evaluation
[Add table with IoU and F1 scores for each class]

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory:**
   - Reduce `CROP_SIZE` in image preparation
   - Decrease batch size in training notebooks
   - Use gradient accumulation

2. **Black Mask Detection:**
   - Ensure proper color mapping in COCO annotations
   - Check `CLASS_COLORS` dictionary matches your annotation scheme

3. **Misaligned Snippets:**
   - Verify all input images have the same resolution
   - Check that Sobol sampling parameters are consistent

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{schnorr2025appearance,
  title={[Your Paper Title]},
  author={Schnorr, Nils and Leimkühler, Thomas},
  booktitle={Proceedings of the International Conference on Computer Applications in Archaeology (CAA)},
  year={2025},
  address={Athens, Greece},
  doi={[TO BE FILLED]}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.




## Acknowledgments

[TO BE FILLED - Add acknowledgments to funding sources, institutions, or individuals who contributed]

---

**Last Updated:** August 2025
**Repository:** https://github.com/NilsSchnorr/AppearanceMeetsGeometry
