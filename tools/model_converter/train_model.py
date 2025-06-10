#!/usr/bin/env python3
"""
Smart Meter Reader OCR - Model Training Pipeline

This script implements a complete training pipeline for digit recognition models.
It supports training CNN models for individual digit recognition and sequence models
for complete meter reading recognition.

Features:
- Data loading and preprocessing
- Model architecture definition (CNN, ResNet, EfficientNet)
- Training with validation monitoring
- Model evaluation and metrics
- TensorFlow Lite conversion for ESP32 deployment
- Comprehensive logging and visualization
- Hyperparameter optimization support

Usage:
    # Train basic CNN model
    python train_model.py --data-dir ./training_data --model-type cnn --epochs 50

    # Train with advanced architecture
    python train_model.py --data-dir ./training_data --model-type efficientnet --epochs 100 --batch-size 32

    # Resume training from checkpoint
    python train_model.py --data-dir ./training_data --resume-from ./checkpoints/model_epoch_20.h5

Author: Frank Amoah A.K.A SiRWaTT Smart Meter Reader OCR Team Lead
"""

import argparse
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from datetime import datetime
from typing import Tuple, List, Dict, Optional, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2
import wandb  # For experiment tracking (optional)
from tqdm import tqdm
import warnings

# Suppress TensorFlow warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MeterOCRTrainer:
    """Complete training pipeline for Smart Meter OCR models"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the trainer with configuration
        
        Args:
            config (Dict[str, Any]): Training configuration dictionary
        """
        self.config = config
        self.model = None
        self.history = None
        
        # Set up paths
        self.data_dir = Path(config['data_dir'])
        self.output_dir = Path(config.get('output_dir', './training_output'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.models_dir = self.output_dir / 'models'
        self.checkpoints_dir = self.output_dir / 'checkpoints'
        self.logs_dir = self.output_dir / 'logs'
        self.plots_dir = self.output_dir / 'plots'
        
        for dir_path in [self.models_dir, self.checkpoints_dir, self.logs_dir, self.plots_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Set random seeds for reproducibility
        np.random.seed(config.get('random_seed', 42))
        tf.random.set_seed(config.get('random_seed', 42))
        
        # Configure GPU if available
        self.setup_gpu()
        
        # Initialize Weights & Biases if enabled
        if config.get('use_wandb', False):
            self.init_wandb()
        
        logger.info(f"Trainer initialized. Output directory: {self.output_dir}")

    def setup_gpu(self):
        """Configure GPU settings for optimal training"""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Enable memory growth to avoid allocating all GPU memory at once
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Found {len(gpus)} GPU(s). Memory growth enabled.")
            except RuntimeError as e:
                logger.warning(f"GPU setup error: {e}")
        else:
            logger.info("No GPU found. Training will use CPU.")

    def init_wandb(self):
        """Initialize Weights & Biases for experiment tracking"""
        try:
            import wandb
            wandb.init(
                project="smart-meter-ocr",
                config=self.config,
                name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            logger.info("Weights & Biases initialized")
        except ImportError:
            logger.warning("Weights & Biases not available. Install with: pip install wandb")
        except Exception as e:
            logger.warning(f"Failed to initialize Weights & Biases: {e}")

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess training data
        
        Returns:
            Tuple containing (X_train, X_val, y_train, y_val)
        """
        logger.info("Loading training data...")
        
        # Load data splits
        train_file = self.data_dir / 'training_export' / 'train.json'
        val_file = self.data_dir / 'training_export' / 'validation.json'
        
        if not train_file.exists():
            raise FileNotFoundError(f"Training data not found: {train_file}")
        
        # Load training data
        with open(train_file, 'r') as f:
            train_data = json.load(f)
            
        # Load validation data if available, otherwise split from training
        if val_file.exists():
            with open(val_file, 'r') as f:
                val_data = json.load(f)
        else:
            logger.info("No separate validation set found. Splitting from training data.")
            train_data, val_data = train_test_split(
                train_data, test_size=0.2, random_state=self.config.get('random_seed', 42)
            )
        
        # Process training data
        X_train, y_train = self.process_data_batch(train_data, is_training=True)
        X_val, y_val = self.process_data_batch(val_data, is_training=False)
        
        logger.info(f"Data loaded - Train: {len(X_train)}, Validation: {len(X_val)}")
        logger.info(f"Input shape: {X_train.shape[1:]}")
        logger.info(f"Number of classes: {len(np.unique(y_train))}")
        
        return X_train, X_val, y_train, y_val

    def process_data_batch(self, data_list: List[Dict], is_training: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a batch of data samples
        
        Args:
            data_list (List[Dict]): List of data samples
            is_training (bool): Whether this is training data (for augmentation)
            
        Returns:
            Tuple of (images, labels)
        """
        images = []
        labels = []
        
        # Setup data augmentation for training
        if is_training and self.config.get('use_augmentation', True):
            augmentation = self.get_augmentation_pipeline()
        else:
            augmentation = None
        
        for sample in tqdm(data_list, desc="Processing data"):
            try:
                # Load image
                image_path = sample['image_path']
                image = cv2.imread(image_path)
                if image is None:
                    logger.warning(f"Could not load image: {image_path}")
                    continue
                
                # Convert to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Extract individual digits from the reading
                reading = sample['label']
                digit_images = self.extract_digit_images(image, reading)
                
                for digit_image, digit_label in digit_images:
                    # Resize to model input size
                    target_size = self.config.get('input_size', (28, 28))
                    digit_image = cv2.resize(digit_image, target_size)
                    
                    # Apply augmentation if training
                    if augmentation is not None:
                        augmented = augmentation(image=digit_image)
                        digit_image = augmented['image']
                    
                    # Normalize pixel values
                    digit_image = digit_image.astype(np.float32) / 255.0
                    
                    # Add channel dimension if grayscale
                    if len(digit_image.shape) == 2:
                        digit_image = np.expand_dims(digit_image, axis=-1)
                    
                    images.append(digit_image)
                    labels.append(int(digit_label))
                    
            except Exception as e:
                logger.warning(f"Error processing sample {sample.get('image_path', 'unknown')}: {e}")
                continue
        
        if not images:
            raise ValueError("No valid images found in data")
            
        return np.array(images), np.array(labels)

    def extract_digit_images(self, image: np.ndarray, reading: str) -> List[Tuple[np.ndarray, str]]:
        """
        Extract individual digit images from a meter reading image
        
        Args:
            image (np.ndarray): Input image
            reading (str): The correct reading for labeling
            
        Returns:
            List of (digit_image, digit_label) tuples
        """
        # This is a simplified implementation
        # In practice, you would use the image processing functions to segment digits
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours (potential digits)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and sort contours by x-coordinate (left to right)
        digit_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Filter by size (reasonable digit dimensions)
            if w > 10 and h > 15 and w < image.shape[1] // 2 and h < image.shape[0] // 2:
                digit_contours.append((x, y, w, h))
        
        # Sort by x-coordinate
        digit_contours.sort(key=lambda x: x[0])
        
        # Extract digit images
        digit_images = []
        for i, (x, y, w, h) in enumerate(digit_contours):
            if i < len(reading):  # Don't extract more digits than we have labels for
                digit_roi = gray[y:y+h, x:x+w]
                digit_label = reading[i]
                digit_images.append((digit_roi, digit_label))
        
        # If we couldn't segment properly, create synthetic examples
        if len(digit_images) == 0 and len(reading) > 0:
            # Fallback: divide image into equal parts
            height, width = gray.shape
            digit_width = width // len(reading)
            
            for i, digit_char in enumerate(reading):
                x_start = i * digit_width
                x_end = min((i + 1) * digit_width, width)
                digit_roi = gray[:, x_start:x_end]
                digit_images.append((digit_roi, digit_char))
        
        return digit_images

    def get_augmentation_pipeline(self) -> A.Compose:
        """
        Create data augmentation pipeline
        
        Returns:
            Albumentations augmentation pipeline
        """
        return A.Compose([
            # Geometric transformations
            A.Rotate(limit=5, p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=5, p=0.3),
            
            # Noise and blur
            A.GaussNoise(var_limit=(10, 50), p=0.2),
            A.GaussianBlur(blur_limit=3, p=0.1),
            
            # Lighting and contrast
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.CLAHE(clip_limit=2.0, p=0.1),
            
            # Pixel-level transformations
            A.OneOf([
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.5),
            ], p=0.2),
        ])

    def build_model(self) -> keras.Model:
        """
        Build the neural network model based on configuration
        
        Returns:
            Compiled Keras model
        """
        model_type = self.config.get('model_type', 'cnn')
        input_shape = (*self.config.get('input_size', (28, 28)), 1)  # Assuming grayscale
        num_classes = 10  # Digits 0-9
        
        if model_type == 'cnn':
            model = self.build_cnn_model(input_shape, num_classes)
        elif model_type == 'resnet':
            model = self.build_resnet_model(input_shape, num_classes)
        elif model_type == 'efficientnet':
            model = self.build_efficientnet_model(input_shape, num_classes)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Compile model
        optimizer = self.get_optimizer()
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"Model built: {model_type}")
        logger.info(f"Total parameters: {model.count_params():,}")
        
        return model

    def build_cnn_model(self, input_shape: Tuple[int, int, int], num_classes: int) -> keras.Model:
        """Build a basic CNN model"""
        model = models.Sequential([
            # First conv block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second conv block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third conv block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            
            # Classifier
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model

    def build_resnet_model(self, input_shape: Tuple[int, int, int], num_classes: int) -> keras.Model:
        """Build a ResNet-inspired model"""
        inputs = layers.Input(shape=input_shape)
        
        # Initial conv layer
        x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Residual blocks
        for filters in [32, 64, 128]:
            x = self.residual_block(x, filters)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Dropout(0.25)(x)
        
        # Global average pooling and classifier
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        return keras.Model(inputs, outputs)

    def residual_block(self, x: tf.Tensor, filters: int) -> tf.Tensor:
        """Create a residual block"""
        shortcut = x
        
        # First conv
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Second conv
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Adjust shortcut if needed
        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, (1, 1), padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        
        # Add shortcut
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        
        return x

    def build_efficientnet_model(self, input_shape: Tuple[int, int, int], num_classes: int) -> keras.Model:
        """Build an EfficientNet-inspired model"""
        # Use EfficientNetB0 as base
        base_model = tf.keras.applications.EfficientNetB0(
            input_shape=input_shape,
            include_top=False,
            weights=None  # Train from scratch for small images
        )
        
        # Add custom classifier
        inputs = base_model.input
        x = base_model.layers[-2].output  # Before final pooling
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        return keras.Model(inputs, outputs)

    def get_optimizer(self) -> keras.optimizers.Optimizer:
        """Get optimizer based on configuration"""
        optimizer_name = self.config.get('optimizer', 'adam')
        learning_rate = self.config.get('learning_rate', 0.001)
        
        if optimizer_name.lower() == 'adam':
            return optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name.lower() == 'sgd':
            return optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer_name.lower() == 'rmsprop':
            return optimizers.RMSprop(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def get_callbacks(self) -> List[keras.callbacks.Callback]:
        """Create training callbacks"""
        callbacks_list = []
        
        # Model checkpointing
        checkpoint_path = self.checkpoints_dir / "model_epoch_{epoch:02d}_val_acc_{val_accuracy:.4f}.h5"
        callbacks_list.append(
            callbacks.ModelCheckpoint(
                str(checkpoint_path),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
        )
        
        # Early stopping
        if self.config.get('early_stopping', True):
            callbacks_list.append(
                callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=self.config.get('patience', 10),
                    restore_best_weights=True,
                    verbose=1
                )
            )
        
        # Learning rate reduction
        callbacks_list.append(
            callbacks.ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        )
        
        # TensorBoard logging
        log_dir = self.logs_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        callbacks_list.append(
            callbacks.TensorBoard(
                log_dir=str(log_dir),
                histogram_freq=1,
                write_graph=True,
                write_images=True
            )
        )
        
        # Custom callback for metrics logging
        callbacks_list.append(MetricsLogger(self.config.get('use_wandb', False)))
        
        return callbacks_list

    def train(self) -> keras.Model:
        """
        Train the model
        
        Returns:
            Trained model
        """
        logger.info("Starting model training...")
        
        # Load data
        X_train, X_val, y_train, y_val = self.load_data()
        
        # Build model
        self.model = self.build_model()
        
        # Print model summary
        self.model.summary()
        
        # Get callbacks
        callbacks_list = self.get_callbacks()
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=self.config.get('batch_size', 32),
            epochs=self.config.get('epochs', 50),
            validation_data=(X_val, y_val),
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Save final model
        final_model_path = self.models_dir / "final_model.h5"
        self.model.save(str(final_model_path))
        logger.info(f"Final model saved: {final_model_path}")
        
        return self.model

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model performance
        
        Args:
            X_test: Test images
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        logger.info("Evaluating model...")
        
        # Get predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Create visualizations
        self.plot_training_history()
        self.plot_confusion_matrix(conf_matrix)
        self.plot_sample_predictions(X_test, y_test, y_pred, y_pred_proba)
        
        evaluation_results = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist()
        }
        
        # Save evaluation results
        eval_file = self.output_dir / "evaluation_results.json"
        with open(eval_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"Evaluation results saved: {eval_file}")
        
        return evaluation_results

    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_confusion_matrix(self, conf_matrix: np.ndarray):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=range(10), yticklabels=range(10))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig(self.plots_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_sample_predictions(self, X_test: np.ndarray, y_test: np.ndarray, 
                              y_pred: np.ndarray, y_pred_proba: np.ndarray):
        """Plot sample predictions"""
        # Select a few random samples
        n_samples = min(16, len(X_test))
        indices = np.random.choice(len(X_test), n_samples, replace=False)
        
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.ravel()
        
        for i, idx in enumerate(indices):
            # Display image
            img = X_test[idx]
            if img.shape[-1] == 1:  # Grayscale
                axes[i].imshow(img.squeeze(), cmap='gray')
            else:
                axes[i].imshow(img)
            
            # Add prediction information
            true_label = y_test[idx]
            pred_label = y_pred[idx]
            confidence = y_pred_proba[idx][pred_label]
            
            color = 'green' if true_label == pred_label else 'red'
            axes[i].set_title(f'True: {true_label}, Pred: {pred_label}\\nConf: {confidence:.3f}', 
                            color=color, fontsize=10)
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'sample_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()

    def convert_to_tflite(self, model_path: Optional[str] = None) -> str:
        """
        Convert trained model to TensorFlow Lite format for ESP32 deployment
        
        Args:
            model_path: Path to model file (if None, uses current model)
            
        Returns:
            Path to converted TFLite model file
        """
        if model_path is not None:
            model = keras.models.load_model(model_path)
        elif self.model is not None:
            model = self.model
        else:
            raise ValueError("No model available for conversion")
        
        logger.info("Converting model to TensorFlow Lite...")
        
        # Create TFLite converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Set optimization flags
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Quantization for smaller model size
        if self.config.get('quantize_model', True):
            # Representative dataset for quantization
            def representative_data_gen():
                # Use a subset of training data for calibration
                X_train, _, _, _ = self.load_data()
                for i in range(min(100, len(X_train))):
                    yield [X_train[i:i+1].astype(np.float32)]
            
            converter.representative_dataset = representative_data_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
        
        # Convert model
        tflite_model = converter.convert()
        
        # Save TFLite model
        tflite_path = self.models_dir / "digit_recognition_model.tflite"
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        
        # Save model info
        model_info = {
            "input_shape": model.input_shape,
            "output_shape": model.output_shape,
            "model_size_bytes": len(tflite_model),
            "quantized": self.config.get('quantize_model', True),
            "conversion_date": datetime.now().isoformat()
        }
        
        info_path = self.models_dir / "model_info.json"
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2, default=str)
        
        logger.info(f"TensorFlow Lite model saved: {tflite_path}")
        logger.info(f"Model size: {len(tflite_model) / 1024:.1f} KB")
        
        return str(tflite_path)


class MetricsLogger(keras.callbacks.Callback):
    """Custom callback for logging metrics"""
    
    def __init__(self, use_wandb: bool = False):
        super().__init__()
        self.use_wandb = use_wandb
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # Log to console
        logger.info(f"Epoch {epoch + 1} - "
                   f"loss: {logs.get('loss', 0):.4f} - "
                   f"accuracy: {logs.get('accuracy', 0):.4f} - "
                   f"val_loss: {logs.get('val_loss', 0):.4f} - "
                   f"val_accuracy: {logs.get('val_accuracy', 0):.4f}")
        
        # Log to Weights & Biases
        if self.use_wandb:
            try:
                import wandb
                wandb.log(logs, step=epoch)
            except ImportError:
                pass


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train Smart Meter OCR Model")
    
    # Data arguments
    parser.add_argument("--data-dir", required=True, help="Path to training data directory")
    parser.add_argument("--output-dir", default="./training_output", help="Output directory for models and logs")
    
    # Model arguments
    parser.add_argument("--model-type", choices=["cnn", "resnet", "efficientnet"], default="cnn",
                       help="Type of model architecture")
    parser.add_argument("--input-size", nargs=2, type=int, default=[28, 28],
                       help="Input image size (width height)")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--optimizer", choices=["adam", "sgd", "rmsprop"], default="adam",
                       help="Optimizer type")
    
    # Regularization and augmentation
    parser.add_argument("--use-augmentation", action="store_true", default=True,
                       help="Use data augmentation")
    parser.add_argument("--early-stopping", action="store_true", default=True,
                       help="Use early stopping")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    
    # Model conversion
    parser.add_argument("--convert-tflite", action="store_true", default=True,
                       help="Convert model to TensorFlow Lite")
    parser.add_argument("--quantize-model", action="store_true", default=True,
                       help="Quantize model for smaller size")
    
    # Experiment tracking
    parser.add_argument("--use-wandb", action="store_true", help="Use Weights & Biases for tracking")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility")
    
    # Resume training
    parser.add_argument("--resume-from", help="Path to model checkpoint to resume from")
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()
    
    # Create configuration dictionary
    config = {
        'data_dir': args.data_dir,
        'output_dir': args.output_dir,
        'model_type': args.model_type,
        'input_size': tuple(args.input_size),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'optimizer': args.optimizer,
        'use_augmentation': args.use_augmentation,
        'early_stopping': args.early_stopping,
        'patience': args.patience,
        'convert_tflite': args.convert_tflite,
        'quantize_model': args.quantize_model,
        'use_wandb': args.use_wandb,
        'random_seed': args.random_seed,
        'resume_from': args.resume_from
    }
    
    try:
        # Initialize trainer
        trainer = MeterOCRTrainer(config)
        
        # Resume from checkpoint if specified
        if args.resume_from:
            logger.info(f"Resuming training from: {args.resume_from}")
            trainer.model = keras.models.load_model(args.resume_from)
        
        # Train model
        model = trainer.train()
        
        # Load test data if available for evaluation
        test_file = Path(args.data_dir) / 'training_export' / 'test.json'
        if test_file.exists():
            logger.info("Loading test data for evaluation...")
            with open(test_file, 'r') as f:
                test_data = json.load(f)
            X_test, y_test = trainer.process_data_batch(test_data, is_training=False)
            trainer.evaluate_model(X_test, y_test)
        else:
            logger.info("No test data found. Skipping final evaluation.")
        
        # Convert to TensorFlow Lite
        if args.convert_tflite:
            tflite_path = trainer.convert_to_tflite()
            logger.info(f"Model ready for ESP32 deployment: {tflite_path}")
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
