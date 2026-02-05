# Deep Learning Transfer Learning with AlexNet

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://www.tensorflow.org/)

A comprehensive implementation of transfer learning techniques using AlexNet on the EMNIST digit dataset. This project demonstrates both **depth-augmented** and **width-augmented** architectures for improved performance on small datasets.

## 🎯 Project Overview

This project explores transfer learning by leveraging a pre-trained AlexNet model to improve digit classification performance on a limited dataset (1,000 digits). The implementation compares three different approaches:

1. **Baseline CNN** - A small custom CNN trained from scratch
2. **Depth-Augmented Model** - Pre-trained AlexNet features fed into a custom CNN
3. **Width-Augmented Model** - Parallel branches combining AlexNet features with direct input processing

## ✨ Key Features

- 🔄 **Transfer Learning**: Utilizes pre-trained AlexNet weights for feature extraction
- 🏗️ **Multiple Architectures**: Implements both depth and width augmentation strategies
- 📊 **Comprehensive Evaluation**: Includes confusion matrices and learning curves
- 🎓 **Educational**: Well-documented code with clear hyperparameter choices
- 📈 **Performance Tracking**: Automated plotting of training metrics

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- TensorFlow 2.x
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

### Installation

1. Clone the repository:
```bash
git clone https://github.com/patrickjcraig/DL-HW5.git
cd DL-HW5
```

2. Install required dependencies:
```bash
pip install tensorflow numpy scikit-learn matplotlib seaborn
```

3. Ensure you have the following files:
   - `digits.npz` - EMNIST digit dataset
   - `TensorFlow/AlexNet_pretrained.h5` - Pre-trained AlexNet weights

### Running the Project

Execute the main training script:
```bash
python main.py
```

This will:
- Load and preprocess the EMNIST digit dataset
- Train the baseline CNN model
- Train the depth-augmented model
- Train the width-augmented model
- Generate learning curves and confusion matrices for all models
- Display comparative test accuracies

## 📁 Project Structure

```
DL-HW5/
├── main.py                          # Main training and evaluation script
├── utils.py                         # Utility functions (data loading, splitting)
├── TensorFlow/
│   └── networks_tf.py              # Neural network architectures
├── digits.npz                      # EMNIST digit dataset (not included)
└── TensorFlow/AlexNet_pretrained.h5 # Pre-trained weights (not included)
```

## 🧠 Model Architectures

### Baseline CNN (`Patricks_Smallish_CNN`)
A lightweight CNN with:
- 2 convolutional layers (8 and 16 filters)
- Fully connected layers (128 neurons)
- Output layer (10 classes)

### Depth-Augmented Model
Architecture flow:
```
Input (28×28×1) → AlexNet (frozen) → Dense Layers → Reshape → Baseline CNN → Output
```

### Width-Augmented Model
Parallel architecture:
```
Input (28×28×1) ─┬─→ AlexNet Branch (frozen) ─┐
                 │                             ├─→ Concatenate → Reshape → CNN → Output
                 └─→ Dense Branch ────────────┘
```

## 📊 Training Configuration

### Dataset Split
- **Training Set**: 500 samples (50.0%)
- **Validation Set**: 200 samples (20.0%)
- **Test Set**: 300 samples (30.0%)

### Hyperparameters
- **Optimizer**: Adam (learning rate: 1e-4)
- **Loss Function**: Sparse Categorical Crossentropy
- **Batch Size**: 32
- **Max Epochs**: 50
- **Early Stopping**: Patience of 5 epochs on validation loss

## 📈 Results

The project generates comprehensive visualizations including:
- **Learning Curves**: Training and validation accuracy/loss over epochs
- **Confusion Matrices**: Per-class performance evaluation for each model
- **Comparative Metrics**: Test accuracies across all three architectures

### Expected Outcomes
- Baseline CNN performance on limited data
- Improved accuracy through transfer learning
- Insights into depth vs. width augmentation effectiveness

## 🔍 Key Insights

### Transfer Learning Benefits
- Pre-trained features reduce overfitting on small datasets
- Frozen AlexNet layers act as powerful feature extractors
- Fine-tuning only the classifier layers speeds up training

### Architecture Comparisons
- **Depth augmentation** adds layers sequentially for hierarchical feature processing
- **Width augmentation** processes features in parallel for diverse representations
- Both approaches leverage pre-trained knowledge effectively

## 📝 Implementation Details

### Data Preprocessing
```python
# Normalization
inputs = inputs / 255.0

# Channel expansion for grayscale images
inputs = np.expand_dims(inputs, axis=-1)
```

### AlexNet Adaptation
The pre-trained AlexNet is modified by:
1. Removing the last 5 layers (classifier head)
2. Freezing all remaining convolutional layers
3. Adding custom dense layers or connecting to a custom CNN

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Patrick J. Craig**

## 🙏 Acknowledgments

- EMNIST dataset from the NIST Special Database
- AlexNet architecture inspired by the original ImageNet paper
- TensorFlow/Keras for the deep learning framework

## 📚 References

- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet Classification with Deep Convolutional Neural Networks
- EMNIST: An extension of MNIST to handwritten letters
- Transfer Learning techniques in Deep Neural Networks

---

**Note**: This project is part of a deep learning course assignment (HW5) focused on understanding and implementing transfer learning techniques.
