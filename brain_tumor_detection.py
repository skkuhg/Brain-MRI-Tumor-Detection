"""
Brain MRI Tumor Detection using Lightweight CNNs

This script implements three lightweight Convolutional Neural Networks (CNNs) 
for brain tumor detection using MRI images. It includes comprehensive data 
visualization and model comparison analysis.

Author: Brain MRI Tumor Detection Project
License: MIT
"""

# Core libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Deep Learning libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Image processing
import cv2
from PIL import Image

# Machine Learning metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
DATA_PATH = "brain_mri_data"  # Update this to your dataset location
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

def main():
    """Main function to run the brain tumor detection analysis"""
    print("🧠 Brain MRI Tumor Detection using Lightweight CNNs")
    print("=" * 60)
    
    print("TensorFlow version:", tf.__version__)
    print("GPU Available:", tf.config.list_physical_devices('GPU'))
    
    # Check dataset
    if not os.path.exists(DATA_PATH):
        print(f"❌ Dataset not found at: {DATA_PATH}")
        print("Please update DATA_PATH to point to your dataset location")
        return
    
    print(f"✅ Dataset found at: {DATA_PATH}")
    
    # Load and preprocess data
    print("\n📊 Loading and preprocessing data...")
    X, y, class_names = load_and_preprocess_data(DATA_PATH, IMAGE_SIZE)
    
    if X is None:
        print("❌ Failed to load dataset")
        return
    
    print(f"Loaded {len(X)} images")
    print(f"Classes: {class_names}")
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=np.argmax(y, axis=1)
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, 
        stratify=np.argmax(y_temp, axis=1)
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Create models
    print("\n🏗️ Creating models...")
    num_classes = len(class_names)
    input_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    
    models = {
        'Custom CNN': create_lightweight_cnn_v1(input_shape, num_classes),
        'MobileNetV2': create_mobilenet_v2_model(input_shape, num_classes),
        'EfficientNetB0': create_efficientnet_model(input_shape, num_classes)
    }
    
    # Train and evaluate models
    print("\n🚀 Training models...")
    histories = {}
    evaluation_results = {}
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        
        # Train model
        history = train_model(model, model_name, X_train, y_train, X_val, y_val)
        histories[model_name] = history
        
        # Evaluate model
        results = evaluate_model(model, model_name, X_test, y_test)
        evaluation_results[model_name] = results
        
        print(f"✅ {model_name} completed!")
        print(f"   Test Accuracy: {results['accuracy']:.4f}")
    
    # Compare results
    print("\n📈 Model Comparison:")
    print("-" * 60)
    print(f"{'Model':<15} {'Accuracy':<10} {'Parameters':<12} {'Size (MB)':<10}")
    print("-" * 60)
    
    for model_name in models.keys():
        acc = evaluation_results[model_name]['accuracy']
        params = models[model_name].count_params()
        size_mb = params * 4 / (1024**2)
        print(f"{model_name:<15} {acc:<10.4f} {params:<12,} {size_mb:<10.2f}")
    
    print("\n🎉 Analysis completed!")
    print("\nFor detailed visualizations and analysis, please run the Jupyter notebook.")

def load_and_preprocess_data(data_path, image_size=(224, 224)):
    """Load and preprocess MRI images from the dataset"""
    try:
        images = []
        labels = []
        class_names = []
        
        # Get class names
        for class_name in os.listdir(data_path):
            class_path = os.path.join(data_path, class_name)
            if os.path.isdir(class_path):
                class_names.append(class_name)
        
        class_names.sort()  # Ensure consistent ordering
        print(f"Classes found: {class_names}")
        
        # Load images and labels
        for class_idx, class_name in enumerate(class_names):
            class_path = os.path.join(data_path, class_name)
            
            for filename in os.listdir(class_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    try:
                        # Load and preprocess image
                        img_path = os.path.join(class_path, filename)
                        img = cv2.imread(img_path)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, image_size)
                        img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
                        
                        images.append(img)
                        labels.append(class_idx)
                        
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
        
        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)
        
        # Convert labels to categorical
        y_categorical = to_categorical(y, len(class_names))
        
        return X, y_categorical, class_names
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None, None

def create_lightweight_cnn_v1(input_shape, num_classes):
    """Create a lightweight CNN model with minimal parameters"""
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fourth convolutional block
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        
        # Dense layers
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model

def create_mobilenet_v2_model(input_shape, num_classes):
    """Create a lightweight model based on MobileNetV2 architecture"""
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        alpha=0.75
    )
    
    base_model.trainable = True
    
    # Fine-tune from this layer onwards
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model

def create_efficientnet_model(input_shape, num_classes):
    """Create a lightweight model based on EfficientNetB0"""
    base_model = EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    base_model.trainable = True
    
    # Fine-tune from this layer onwards
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.1),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model

def train_model(model, model_name, X_train, y_train, X_val, y_val):
    """Train a model and return training history"""
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=0
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-7,
            verbose=0
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=0
    )
    
    return history

def evaluate_model(model, model_name, X_test, y_test):
    """Evaluate model performance on test set"""
    test_predictions = model.predict(X_test, verbose=0)
    test_pred_classes = np.argmax(test_predictions, axis=1)
    test_true_classes = np.argmax(y_test, axis=1)
    
    test_accuracy = accuracy_score(test_true_classes, test_pred_classes)
    test_precision = precision_score(test_true_classes, test_pred_classes, average='weighted')
    test_recall = recall_score(test_true_classes, test_pred_classes, average='weighted')
    test_f1 = f1_score(test_true_classes, test_pred_classes, average='weighted')
    
    results = {
        'accuracy': test_accuracy,
        'precision': test_precision,
        'recall': test_recall,
        'f1_score': test_f1,
        'predictions': test_predictions,
        'pred_classes': test_pred_classes,
        'true_classes': test_true_classes
    }
    
    return results

if __name__ == "__main__":
    main()