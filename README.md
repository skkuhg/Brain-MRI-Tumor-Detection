# Brain MRI Tumor Detection using Lightweight CNNs

A comprehensive implementation of three lightweight Convolutional Neural Networks for brain tumor detection using MRI images, featuring detailed data visualization and model comparison analysis.

## 🧠 Project Overview

This project implements and compares three different lightweight CNN architectures for brain tumor detection:

1. **Custom Lightweight CNN** - A from-scratch architecture optimized for efficiency
2. **MobileNetV2-based** - Utilizing depthwise separable convolutions
3. **EfficientNetB0-based** - Employing compound scaling for optimal performance

## 🎯 Key Features

- **Comprehensive Data Analysis**: Complete dataset exploration and visualization
- **Three Model Architectures**: Comparison of different lightweight CNN approaches
- **Performance Metrics**: Detailed evaluation using accuracy, precision, recall, and F1-score
- **Data Visualization**: Over 10 different types of plots and charts
- **Model Efficiency Analysis**: Accuracy vs model size trade-offs
- **Professional Visualizations**: Confusion matrices, training curves, and prediction analysis

## 📊 Dataset Structure

The dataset should be organized as follows:
```
brain_mri_data/
├── no/          # MRI images without tumor
└── yes/         # MRI images with tumor
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional but recommended)

### Required Libraries
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install tensorflow opencv-python matplotlib seaborn pandas numpy scikit-learn pillow jupyter
```

## 🚀 Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/skkuhg/brain-mri-tumor-detection.git
   cd brain-mri-tumor-detection
   ```

2. **Prepare your dataset**:
   - Organize your MRI images in the structure shown above
   - Update the `DATA_PATH` variable in the notebook to point to your dataset location

3. **Run the notebook**:
   - Open `brain_mri_tumor_detection.ipynb` in Jupyter Notebook or VS Code
   - Execute cells sequentially to train and evaluate models

## 📈 Model Performance

The notebook provides comprehensive analysis including:

- **Training History**: Loss and accuracy curves for all models
- **Confusion Matrices**: Visual representation of classification performance
- **Performance Comparison**: Side-by-side metrics comparison
- **Efficiency Analysis**: Parameter count vs accuracy trade-offs
- **Prediction Visualization**: Sample predictions with confidence scores

## 🏗️ Model Architectures

### Custom Lightweight CNN
- **Parameters**: ~500K parameters
- **Features**: Global Average Pooling, Batch Normalization, Dropout
- **Optimization**: Minimal parameters while maintaining performance

### MobileNetV2-based
- **Parameters**: ~2.3M parameters
- **Features**: Depthwise separable convolutions, Transfer learning
- **Optimization**: Pre-trained weights with fine-tuning

### EfficientNetB0-based
- **Parameters**: ~4M parameters
- **Features**: Compound scaling, Advanced architecture
- **Optimization**: State-of-the-art efficiency

## 📋 Notebook Contents

1. **Import Libraries** - All necessary dependencies
2. **Data Loading** - Dataset exploration and structure analysis
3. **Data Preprocessing** - Image normalization and augmentation
4. **Data Visualization** - Comprehensive dataset analysis
5. **Model 1: Custom CNN** - Lightweight architecture from scratch
6. **Model 2: MobileNetV2** - Transfer learning approach
7. **Model 3: EfficientNetB0** - Advanced lightweight model
8. **Training & Evaluation** - Model training with callbacks
9. **Performance Comparison** - Detailed metrics analysis
10. **Results Visualization** - Predictions and confidence analysis

## 🔬 Technical Details

### Data Augmentation
- Rotation (±15°)
- Width/Height shift (±10%)
- Horizontal flip
- Zoom (±10%)

### Training Configuration
- **Epochs**: 20 (with early stopping)
- **Batch Size**: 32
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Callbacks**: Early Stopping, Learning Rate Reduction

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Model Efficiency (Parameters vs Performance)

## 📊 Expected Results

The models typically achieve:
- **Custom CNN**: 85-90% accuracy with minimal parameters
- **MobileNetV2**: 90-95% accuracy with good efficiency
- **EfficientNetB0**: 92-97% accuracy with optimal performance

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🔗 Repository Structure

```
brain-mri-tumor-detection/
├── README.md
├── requirements.txt
├── LICENSE
├── brain_mri_tumor_detection.ipynb
└── .gitignore
```

## 🙏 Acknowledgments

- TensorFlow/Keras team for the deep learning framework
- MobileNet and EfficientNet authors for the architectures
- Medical imaging community for advancing brain tumor detection research

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note**: This project is for educational and research purposes. Always consult with medical professionals for actual medical diagnosis.