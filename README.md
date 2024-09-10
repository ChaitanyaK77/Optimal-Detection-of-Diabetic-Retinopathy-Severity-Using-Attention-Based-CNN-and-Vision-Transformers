
# Optimal Detection of Diabetic Retinopathy Severity Using Attention-Based CNN and Vision Transformers

This repository contains the implementation of a hybrid model combining **Attention-Based Convolutional Neural Networks (CNNs)** and **Vision Transformers (ViT)** to classify the severity levels of diabetic retinopathy. This approach leverages CNNs to capture local features in retinal images and ViTs to model long-range dependencies, achieving superior performance in multi-class classification tasks.

## Model Overview

Diabetic Retinopathy is a severe eye condition that can lead to blindness if not detected early. In this project, we introduce a robust AI model capable of detecting and classifying retinal images into five severity levels of diabetic retinopathy:

1. **No_DR** (No Diabetic Retinopathy)
2. **Mild**
3. **Moderate**
4. **Severe**
5. **Proliferative_DR** (Advanced Stage)

The hybrid architecture combines the best of both **CNN** and **Vision Transformers**:
- **CNNs** are used to extract local texture and features from retinal images.
- **Vision Transformers (ViT)** are employed for capturing global contextual information through self-attention mechanisms.

## 📁 Project Structure

```
├── /data                     # Contains image datasets for training and validation
├── /notebooks                # Jupyter notebooks for data exploration and preprocessing
├── /models                   # Model training and evaluation scripts
├── /results                  # Logs, checkpoints, and evaluation metrics
├── /utils                    # Utility scripts for data preprocessing and augmentation
└── README.md                 # Project documentation
```

## 📊 Dataset

The dataset contains retinal images labeled according to the severity of diabetic retinopathy. It is split into two sets:
- **Training Set**: `/colored_images`
- **Validation Set**: `/colored_images_split`

Each folder contains subfolders named `0 - No_DR`, `1 - Mild`, `2 - Moderate`, `3 - Severe`, and `4 - Proliferate_DR`.
Dataset can be freely accesed at :"https://www.kaggle.com/competitions/aptos2019-blindness-detection/data"

## Model Architecture

The model is built using Vision Transformers (ViT) and includes the following components:

- **Input**: RGB retinal images
- **Convolutional Layers**: Initial layers for extracting local features from images
- **SEBlock**: Attention mechanism to recalibrate feature maps and focus on the important channels
- **Vision Transformer**: Extracts global information from the input data and refines the representation through multi-headed self-attention
- **Fully Connected Layers**: For classification into one of the five diabetic retinopathy severity levels
- **Output**: Softmax layer for multi-class classification

### Vision Transformer (ViT)

- Uses patch embeddings to split the input image into patches.
- Each patch is passed through a transformer encoder consisting of multi-headed self-attention layers and feed-forward neural networks.
- Provides global contextual information to better understand the relationship between different parts of the image.

### SEBlock

- Squeeze-and-Excitation (SE) Block is integrated into the CNN layers to recalibrate the importance of feature channels dynamically.
- It enhances the network's sensitivity to important features while suppressing less informative ones.

The images are preprocessed and tokenized using `ViTImageProcessor`.

## Requirements

- Python 3.x
- PyTorch
- Transformers (Hugging Face)
- Datasets (Hugging Face)
- PIL (Python Imaging Library)
- CUDA (if using GPU)

## Installation

```bash
pip install torch transformers datasets pillow
```
## Conclusion

This model demonstrates the effectiveness of combining Convolutional Neural Networks (CNN) with Vision Transformers (ViT) and attention mechanisms for detecting diabetic retinopathy severity. With further tuning and larger datasets, this approach can be highly valuable in clinical settings to automate the early detection of DR and assist ophthalmologists in decision-making.

## License

This project is licensed under the MIT License.
