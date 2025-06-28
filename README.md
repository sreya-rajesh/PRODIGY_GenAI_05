# PRODIGY_GenAI_05
# Neural Style Transfer

A Python implementation of neural style transfer using PyTorch and VGG19, allowing you to combine the content of one image with the artistic style of another.

## Overview

This project implements the neural style transfer technique described in the paper "A Neural Algorithm of Artistic Style" by Gatys et al. The algorithm uses a pre-trained VGG19 convolutional neural network to extract content and style features, then optimizes a target image to minimize both content and style losses.

## Features

- **Interactive CLI**: Easy-to-use command-line interface with customizable parameters
- **URL-based image loading**: Load images directly from web URLs
- **GPU acceleration**: Automatic GPU detection and utilization when available
- **Flexible parameters**: Adjustable content weight, style weight, learning rate, and iterations
- **Progress monitoring**: Real-time loss tracking during optimization
- **Image preprocessing**: Automatic resizing and normalization

## Requirements

```
torch>=1.9.0
torchvision>=0.10.0
Pillow>=8.0.0
matplotlib>=3.3.0
requests>=2.25.0
```

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd neural-style-transfer
```

2. Install the required dependencies:
```bash
pip install torch torchvision Pillow matplotlib requests
```

## Usage

### Basic Usage

Run the script and follow the interactive prompts:

```bash
python task5.py
```

The program will ask you to provide:
1. **Content image URL**: The image whose content/structure you want to preserve
2. **Style image URL**: The image whose artistic style you want to apply
3. **Optional parameters**: Various hyperparameters (or use defaults)

### Example

```
Neural Style Transfer
==================================================
Enter the URL of your content image: https://example.com/content.jpg
Enter the URL of your style image: https://example.com/style.jpg

Optional parameters (press Enter for defaults):
Content weight (default: 1e4): 
Style weight (default: 1e6): 
Learning rate (default: 0.01): 
Number of iterations (default: 500): 
Show progress every N iterations (default: 50): 
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Content Weight | 1e4 | Controls how much the output should resemble the content image |
| Style Weight | 1e6 | Controls how much the output should resemble the style image |
| Learning Rate | 0.01 | Step size for the optimization algorithm |
| Iterations | 500 | Number of optimization steps |
| Show Every | 50 | Frequency of progress updates |

### Parameter Tuning Tips

- **Higher content weight**: More faithful to original content structure
- **Higher style weight**: More pronounced artistic style transfer
- **Lower learning rate**: Slower but more stable convergence
- **More iterations**: Better quality but longer processing time

## Architecture

The implementation uses several key components:

### VGGFeatureExtractor
- Based on pre-trained VGG19 network
- Extracts features from multiple convolutional layers
- Content features from: `conv4_2` (layer 21)
- Style features from: `conv1_1, conv2_1, conv3_1, conv4_1, conv5_1` (layers 0, 5, 10, 19, 28)

### Loss Functions
- **Content Loss**: Mean squared error between target and content features
- **Style Loss**: Mean squared error between Gram matrices of target and style features
- **Total Loss**: Weighted combination of content and style losses

### Optimization
- Uses Adam optimizer to iteratively update the target image
- Starts with the content image as initialization
- Applies gradient descent to minimize the combined loss function

## Output

The program displays:
- Real-time optimization progress with loss values
- Final stylized image using matplotlib
- Processing status and error messages

## Performance

- **GPU recommended**: Significantly faster processing with CUDA-enabled GPU
- **Memory usage**: Depends on image size (images are resized to max 512px)
- **Processing time**: Typically 2-10 minutes depending on hardware and parameters

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce image size or batch size
2. **URL loading errors**: Ensure image URLs are accessible and valid
3. **Slow performance**: Check if GPU is being utilized
4. **Poor results**: Try adjusting content/style weight ratios

### Error Messages

- `Error loading content/style image`: Check URL validity and internet connection
- `Using device: cpu`: GPU not available, processing will be slower

## Technical Details

### Image Preprocessing
- Images are normalized using ImageNet statistics
- Automatic resizing while maintaining aspect ratio
- Convert RGB images to tensor format

### Feature Extraction
- Uses VGG19 pre-trained on ImageNet
- Extracts multi-scale features for robust style representation
- Frozen network weights (no training required)

### Gram Matrix
- Captures style information through feature correlations
- Computed as: `G = F * F^T` where F is the feature map
- Normalized by feature map dimensions


