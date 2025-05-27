#!/usr/bin/env python3
"""
Smart Meter Reader OCR - Benchmarking Tool

This script provides comprehensive evaluation and benchmarking capabilities
for trained OCR models. It can test models on various datasets, compare
different model architectures, and generate detailed performance reports.

Features:
- Model accuracy evaluation
- Speed benchmarking
- Error analysis and visualization
- Cross-dataset validation
- Model comparison
- Performance profiling
- Report generation

Usage:
    # Basic evaluation
    python ocr_benchmark.py --model ./models/final_model.h5 --test-data ./test_images

    # Compare multiple models
    python ocr_benchmark.py --compare-models ./models/cnn.h5 ./models/resnet.h5 --test-data ./test_images

    # Generate comprehensive report
    python ocr_benchmark.py --model ./models/final_model.h5 --test-data ./test_images --full-report

Author: Smart Meter Reader OCR Team
"""

import argparse
import os
import json
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import psutil
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
from tqdm import tqdm
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Data class to store benchmark results"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    inference_time_ms: float
    memory_usage_mb: float
    model_size_mb: float
    confusion_matrix: np.ndarray
    classification_report: Dict
    error_analysis: Dict
    performance_by_digit: Dict

class OCRBenchmark:
    """Comprehensive OCR model benchmarking tool"""
    
    def __init__(self, output_dir: str = "./benchmark_results"):
        """
        Initialize the benchmark tool
        
        Args:
            output_dir (str): Directory to save benchmark results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.plots_dir = self.output_dir / "plots"
        self.reports_dir = self.output_dir / "reports"
        self.data_dir = self.output_dir / "data"
        
        for dir_path in [self.plots_dir, self.reports_dir, self.data_dir]:
            dir_path.mkdir(exist_ok=True)
        
        self.benchmark_results = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"Benchmark tool initialized. Output directory: {self.output_dir}")

    def load_test_data(self, test_data_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load test data from various formats
        
        Args:
            test_data_path (str): Path to test data (JSON file or image directory)
            
        Returns:
            Tuple of (images, labels, filenames)
        """
        test_path = Path(test_data_path)
        
        if test_path.is_file() and test_path.suffix == '.json':
            return self._load_from_json(test_path)
        elif test_path.is_dir():
            return self._load_from_directory(test_path)
        else:
            raise ValueError(f"Unsupported test data format: {test_data_path}")

    def _load_from_json(self, json_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load test data from JSON file (training export format)"""
        with open(json_path, 'r') as f:
            test_data = json.load(f)
        
        images = []
        labels = []
        filenames = []
        
        logger.info(f"Loading {len(test_data)} test samples from JSON...")
        
        for sample in tqdm(test_data, desc="Loading test data"):
            try:
                # Load image
                image_path = sample['image_path']
                if not os.path.exists(image_path):
                    logger.warning(f"Image not found: {image_path}")
                    continue
                
                image = cv2.imread(image_path)
                if image is None:
                    logger.warning(f"Could not load image: {image_path}")
                    continue
                
                # Convert to RGB and extract digits
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                reading = sample['label']
                
                # Extract individual digits (simplified implementation)
                digit_images = self._extract_digits_for_testing(image, reading)
                
                for digit_image, digit_label in digit_images:
                    # Resize to standard size
                    digit_image = cv2.resize(digit_image, (28, 28))
                    
                    # Normalize
                    if len(digit_image.shape) == 3:
                        digit_image = cv2.cvtColor(digit_image, cv2.COLOR_RGB2GRAY)
                    
                    digit_image = digit_image.astype(np.float32) / 255.0
                    digit_image = np.expand_dims(digit_image, axis=-1)
                    
                    images.append(digit_image)
                    labels.append(int(digit_label))
                    filenames.append(f"{Path(image_path).stem}_{digit_label}")
                    
            except Exception as e:
                logger.warning(f"Error processing sample: {e}")
                continue
        
        return np.array(images), np.array(labels), filenames

    def _load_from_directory(self, dir_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load test data from directory of labeled images"""
        images = []
        labels = []
        filenames = []
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(dir_path.glob(f"*{ext}")))
            image_files.extend(list(dir_path.glob(f"*{ext.upper()}")))
        
        logger.info(f"Loading {len(image_files)} images from directory...")
        
        for image_file in tqdm(image_files, desc="Loading images"):
            try:
                # Extract label from filename (assuming format: "digit_X_..." or "X_...")
                filename = image_file.stem
                label = None
                
                # Try different naming conventions
                if '_' in filename:
                    parts = filename.split('_')
                    for part in parts:
                        if part.isdigit() and len(part) == 1:
                            label = int(part)
                            break
                elif filename.isdigit() and len(filename) == 1:
                    label = int(filename)
                
                if label is None:
                    logger.warning(f"Could not extract label from filename: {filename}")
                    continue
                
                # Load and preprocess image
                image = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                
                image = cv2.resize(image, (28, 28))
                image = image.astype(np.float32) / 255.0
                image = np.expand_dims(image, axis=-1)
                
                images.append(image)
                labels.append(label)
                filenames.append(filename)
                
            except Exception as e:
                logger.warning(f"Error processing {image_file}: {e}")
                continue
        
        return np.array(images), np.array(labels), filenames

    def _extract_digits_for_testing(self, image: np.ndarray, reading: str) -> List[Tuple[np.ndarray, str]]:
        """Extract individual digits from meter reading image for testing"""
        # This is a simplified implementation for testing
        # In practice, this would use the same segmentation as in training
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        height, width = gray.shape
        
        # Simple approach: divide image into equal segments
        digit_images = []
        digit_width = width // len(reading)
        
        for i, digit_char in enumerate(reading):
            x_start = i * digit_width
            x_end = min((i + 1) * digit_width, width)
            digit_roi = gray[:, x_start:x_end]
            digit_images.append((digit_roi, digit_char))
        
        return digit_images

    def benchmark_model(
        self, 
        model_path: str, 
        test_images: np.ndarray, 
        test_labels: np.ndarray,
        model_name: Optional[str] = None
    ) -> BenchmarkResult:
        """
        Comprehensive benchmark of a single model
        
        Args:
            model_path (str): Path to the model file
            test_images (np.ndarray): Test images
            test_labels (np.ndarray): Test labels  
            model_name (str, optional): Name for the model
            
        Returns:
            BenchmarkResult: Comprehensive benchmark results
        """
        if model_name is None:
            model_name = Path(model_path).stem
        
        logger.info(f"Benchmarking model: {model_name}")
        
        # Load model
        if model_path.endswith('.tflite'):
            model = self._load_tflite_model(model_path)
            is_tflite = True
        else:
            model = tf.keras.models.load_model(model_path)
            is_tflite = False
        
        # Get model size
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        
        # Memory usage before inference
        process = psutil.Process()
        memory_before = process.memory_info().rss / (1024 * 1024)
        
        # Accuracy evaluation
        logger.info("Evaluating accuracy...")
        if is_tflite:
            predictions, inference_times = self._predict_tflite(model, test_images)
        else:
            predictions, inference_times = self._predict_keras(model, test_images)
        
        # Memory usage after inference
        memory_after = process.memory_info().rss / (1024 * 1024)
        memory_usage_mb = memory_after - memory_before
        
        # Calculate metrics
        predicted_labels = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(test_labels, predicted_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels, predicted_labels, average='weighted'
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(test_labels, predicted_labels)
        
        # Classification report
        class_report = classification_report(
            test_labels, predicted_labels, output_dict=True
        )
        
        # Performance analysis
        avg_inference_time = np.mean(inference_times) * 1000  # Convert to ms
        
        # Error analysis
        error_analysis = self._analyze_errors(
            test_labels, predicted_labels, predictions
        )
        
        # Performance by digit
        performance_by_digit = self._analyze_performance_by_digit(
            test_labels, predicted_labels, predictions
        )
        
        result = BenchmarkResult(
            model_name=model_name,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            inference_time_ms=avg_inference_time,
            memory_usage_mb=memory_usage_mb,
            model_size_mb=model_size_mb,
            confusion_matrix=conf_matrix,
            classification_report=class_report,
            error_analysis=error_analysis,
            performance_by_digit=performance_by_digit
        )
        
        self.benchmark_results.append(result)
        
        logger.info(f"Benchmark completed for {model_name}")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Inference time: {avg_inference_time:.2f}ms")
        logger.info(f"  Model size: {model_size_mb:.2f}MB")
        
        return result

    def _load_tflite_model(self, model_path: str):
        """Load TensorFlow Lite model"""
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter

    def _predict_keras(self, model, test_images: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        """Make predictions using Keras model"""
        batch_size = 32
        predictions = []
        inference_times = []
        
        for i in range(0, len(test_images), batch_size):
            batch = test_images[i:i + batch_size]
            
            start_time = time.time()
            batch_pred = model.predict(batch, verbose=0)
            end_time = time.time()
            
            predictions.append(batch_pred)
            inference_times.extend([(end_time - start_time) / len(batch)] * len(batch))
        
        return np.vstack(predictions), inference_times

    def _predict_tflite(self, interpreter, test_images: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        """Make predictions using TensorFlow Lite model"""
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        predictions = []
        inference_times = []
        
        for image in tqdm(test_images, desc="TFLite inference"):
            # Prepare input
            input_data = np.expand_dims(image, axis=0)
            if input_details[0]['dtype'] == np.uint8:
                input_data = (input_data * 255).astype(np.uint8)
            else:
                input_data = input_data.astype(np.float32)
            
            # Run inference
            start_time = time.time()
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            end_time = time.time()
            
            predictions.append(output_data[0])
            inference_times.append(end_time - start_time)
        
        return np.array(predictions), inference_times

    def _analyze_errors(
        self, 
        true_labels: np.ndarray, 
        predicted_labels: np.ndarray, 
        predictions: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze prediction errors"""
        errors = {}
        
        # Find misclassified samples
        misclassified = true_labels != predicted_labels
        error_count = np.sum(misclassified)
        error_rate = error_count / len(true_labels)
        
        errors['total_errors'] = int(error_count)
        errors['error_rate'] = float(error_rate)
        
        # Most common errors
        error_pairs = []
        for true_label, pred_label in zip(true_labels[misclassified], predicted_labels[misclassified]):
            error_pairs.append((int(true_label), int(pred_label)))
        
        from collections import Counter
        common_errors = Counter(error_pairs).most_common(10)
        errors['most_common_errors'] = [
            {'true': true, 'predicted': pred, 'count': count}
            for (true, pred), count in common_errors
        ]
        
        # Low confidence predictions
        confidences = np.max(predictions, axis=1)
        low_confidence_mask = confidences < 0.8
        errors['low_confidence_count'] = int(np.sum(low_confidence_mask))
        errors['avg_confidence'] = float(np.mean(confidences))
        
        return errors

    def _analyze_performance_by_digit(
        self, 
        true_labels: np.ndarray, 
        predicted_labels: np.ndarray, 
        predictions: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Analyze performance for each digit separately"""
        performance = {}
        
        for digit in range(10):
            digit_mask = true_labels == digit
            if np.sum(digit_mask) == 0:
                continue
            
            digit_true = true_labels[digit_mask]
            digit_pred = predicted_labels[digit_mask]
            digit_probs = predictions[digit_mask]
            
            # Calculate metrics for this digit
            accuracy = accuracy_score(digit_true, digit_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                digit_true, digit_pred, average='weighted', zero_division=0
            )
            
            # Average confidence for this digit
            confidence = np.mean(np.max(digit_probs, axis=1))
            
            performance[str(digit)] = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'avg_confidence': float(confidence),
                'sample_count': int(np.sum(digit_mask))
            }
        
        return performance

    def compare_models(
        self, 
        model_paths: List[str], 
        test_images: np.ndarray, 
        test_labels: np.ndarray
    ) -> List[BenchmarkResult]:
        """
        Compare multiple models on the same test set
        
        Args:
            model_paths (List[str]): List of model file paths
            test_images (np.ndarray): Test images
            test_labels (np.ndarray): Test labels
            
        Returns:
            List[BenchmarkResult]: Results for all models
        """
        logger.info(f"Comparing {len(model_paths)} models...")
        
        results = []
        for model_path in model_paths:
            try:
                result = self.benchmark_model(model_path, test_images, test_labels)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to benchmark {model_path}: {e}")
                continue
        
        # Generate comparison plots
        self._plot_model_comparison(results)
        
        return results

    def _plot_model_comparison(self, results: List[BenchmarkResult]):
        """Create comparison plots for multiple models"""
        if len(results) < 2:
            return
        
        # Metrics comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        model_names = [r.model_name for r in results]
        metrics = {
            'Accuracy': [r.accuracy for r in results],
            'Precision': [r.precision for r in results],
            'Recall': [r.recall for r in results],
            'F1-Score': [r.f1_score for r in results]
        }
        
        # Plot metrics
        for i, (metric_name, values) in enumerate(metrics.items()):
            ax = axes[i // 2, i % 2]
            bars = ax.bar(model_names, values)
            ax.set_title(f'{metric_name} Comparison')
            ax.set_ylabel(metric_name)
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'model_comparison_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Performance vs Size trade-off
        plt.figure(figsize=(10, 6))
        accuracies = [r.accuracy for r in results]
        sizes = [r.model_size_mb for r in results]
        
        plt.scatter(sizes, accuracies, s=100, alpha=0.7)
        for i, name in enumerate(model_names):
            plt.annotate(name, (sizes[i], accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('Model Size (MB)')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Model Size Trade-off')
        plt.grid(True, alpha=0.3)
        plt.savefig(self.plots_dir / 'accuracy_vs_size.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Speed vs Accuracy trade-off
        plt.figure(figsize=(10, 6))
        inference_times = [r.inference_time_ms for r in results]
        
        plt.scatter(inference_times, accuracies, s=100, alpha=0.7)
        for i, name in enumerate(model_names):
            plt.annotate(name, (inference_times[i], accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('Inference Time (ms)')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Inference Speed Trade-off')
        plt.grid(True, alpha=0.3)
        plt.savefig(self.plots_dir / 'accuracy_vs_speed.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_visualizations(self, result: BenchmarkResult):
        """Generate visualization plots for benchmark results"""
        
        # Confusion Matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(result.confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=range(10), yticklabels=range(10))
        plt.title(f'Confusion Matrix - {result.model_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig(self.plots_dir / f'confusion_matrix_{result.model_name}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Performance by digit
        if result.performance_by_digit:
            digits = list(result.performance_by_digit.keys())
            accuracies = [result.performance_by_digit[d]['accuracy'] for d in digits]
            
            plt.figure(figsize=(12, 6))
            bars = plt.bar(digits, accuracies)
            plt.title(f'Accuracy by Digit - {result.model_name}')
            plt.xlabel('Digit')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1)
            
            # Add value labels
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{acc:.3f}', ha='center', va='bottom')
            
            plt.savefig(self.plots_dir / f'accuracy_by_digit_{result.model_name}.png',
                       dpi=300, bbox_inches='tight')
            plt.close()

    def generate_report(self, results: List[BenchmarkResult], save_json: bool = True) -> Dict:
        """
        Generate comprehensive benchmark report
        
        Args:
            results (List[BenchmarkResult]): Benchmark results
            save_json (bool): Whether to save report as JSON
            
        Returns:
            Dict: Complete report data
        """
        logger.info("Generating comprehensive report...")
        
        report = {
            'benchmark_info': {
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'models_tested': len(results),
                'test_samples': len(results[0].confusion_matrix) if results else 0
            },
            'summary': {},
            'detailed_results': [],
            'recommendations': []
        }
        
        if not results:
            logger.warning("No benchmark results to report")
            return report
        
        # Summary statistics
        accuracies = [r.accuracy for r in results]
        inference_times = [r.inference_time_ms for r in results]
        model_sizes = [r.model_size_mb for r in results]
        
        report['summary'] = {
            'best_accuracy': {
                'model': results[np.argmax(accuracies)].model_name,
                'value': float(max(accuracies))
            },
            'fastest_inference': {
                'model': results[np.argmin(inference_times)].model_name,
                'value': float(min(inference_times))
            },
            'smallest_model': {
                'model': results[np.argmin(model_sizes)].model_name,
                'value': float(min(model_sizes))
            },
            'average_accuracy': float(np.mean(accuracies)),
            'accuracy_std': float(np.std(accuracies))
        }
        
        # Detailed results for each model
        for result in results:
            detailed = {
                'model_name': result.model_name,
                'metrics': {
                    'accuracy': result.accuracy,
                    'precision': result.precision,
                    'recall': result.recall,
                    'f1_score': result.f1_score
                },
                'performance': {
                    'inference_time_ms': result.inference_time_ms,
                    'memory_usage_mb': result.memory_usage_mb,
                    'model_size_mb': result.model_size_mb
                },
                'error_analysis': result.error_analysis,
                'performance_by_digit': result.performance_by_digit
            }
            report['detailed_results'].append(detailed)
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(results)
        
        # Save report
        if save_json:
            report_file = self.reports_dir / f"benchmark_report_{self.session_id}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Report saved: {report_file}")
        
        # Generate markdown report
        self._generate_markdown_report(report)
        
        return report

    def _generate_recommendations(self, results: List[BenchmarkResult]) -> List[str]:
        """Generate recommendations based on benchmark results"""
        recommendations = []
        
        if not results:
            return recommendations
        
        # Find best model by different criteria
        best_accuracy_idx = np.argmax([r.accuracy for r in results])
        best_speed_idx = np.argmin([r.inference_time_ms for r in results])
        best_size_idx = np.argmin([r.model_size_mb for r in results])
        
        best_acc_model = results[best_accuracy_idx]
        best_speed_model = results[best_speed_idx]
        best_size_model = results[best_size_idx]
        
        # Accuracy recommendations
        if best_acc_model.accuracy < 0.85:
            recommendations.append(
                "⚠️ Best model accuracy is below 85%. Consider collecting more training data "
                "or using data augmentation to improve performance."
            )
        elif best_acc_model.accuracy > 0.95:
            recommendations.append(
                "✅ Excellent accuracy achieved! Model is ready for production deployment."
            )
        
        # Speed recommendations
        if best_speed_model.inference_time_ms > 100:
            recommendations.append(
                f"⚠️ Inference time is {best_speed_model.inference_time_ms:.1f}ms. "
                "Consider model quantization or pruning for faster inference."
            )
        
        # Size recommendations
        if best_size_model.model_size_mb > 10:
            recommendations.append(
                f"⚠️ Model size is {best_size_model.model_size_mb:.1f}MB. "
                "Consider quantization for ESP32 deployment (recommended <2MB)."
            )
        
        # Error analysis recommendations
        for result in results:
            if result.error_analysis.get('low_confidence_count', 0) > len(results) * 0.1:
                recommendations.append(
                    f"⚠️ {result.model_name} has many low confidence predictions. "
                    "Consider improving data quality or model architecture."
                )
        
        # Performance balance recommendation
        if len(results) > 1:
            # Calculate performance score (weighted accuracy, speed, size)
            scores = []
            for r in results:
                score = (r.accuracy * 0.6 + 
                        (1.0 - min(r.inference_time_ms / 1000, 1.0)) * 0.3 +
                        (1.0 - min(r.model_size_mb / 50, 1.0)) * 0.1)
                scores.append(score)
            
            best_overall_idx = np.argmax(scores)
            recommendations.append(
                f"🎯 Recommended model for deployment: {results[best_overall_idx].model_name} "
                f"(balanced accuracy: {results[best_overall_idx].accuracy:.3f}, "
                f"speed: {results[best_overall_idx].inference_time_ms:.1f}ms, "
                f"size: {results[best_overall_idx].model_size_mb:.1f}MB)"
            )
        
        return recommendations

    def _generate_markdown_report(self, report: Dict):
        """Generate a markdown report"""
        markdown_content = f"""# OCR Model Benchmark Report

**Generated:** {report['benchmark_info']['timestamp']}  
**Session ID:** {report['benchmark_info']['session_id']}  
**Models Tested:** {report['benchmark_info']['models_tested']}

## Summary

| Metric | Best Model | Value |
|--------|------------|-------|
| **Highest Accuracy** | {report['summary']['best_accuracy']['model']} | {report['summary']['best_accuracy']['value']:.4f} |
| **Fastest Inference** | {report['summary']['fastest_inference']['model']} | {report['summary']['fastest_inference']['value']:.2f}ms |
| **Smallest Size** | {report['summary']['smallest_model']['model']} | {report['summary']['smallest_model']['value']:.2f}MB |

**Average Accuracy:** {report['summary']['average_accuracy']:.4f} ± {report['summary']['accuracy_std']:.4f}

## Detailed Results

"""
        
        for result in report['detailed_results']:
            markdown_content += f"""### {result['model_name']}

| Metric | Value |
|--------|-------|
| Accuracy | {result['metrics']['accuracy']:.4f} |
| Precision | {result['metrics']['precision']:.4f} |
| Recall | {result['metrics']['recall']:.4f} |
| F1-Score | {result['metrics']['f1_score']:.4f} |
| Inference Time | {result['performance']['inference_time_ms']:.2f}ms |
| Model Size | {result['performance']['model_size_mb']:.2f}MB |
| Memory Usage | {result['performance']['memory_usage_mb']:.2f}MB |

**Error Analysis:**
- Total Errors: {result['error_analysis']['total_errors']}
- Error Rate: {result['error_analysis']['error_rate']:.4f}
- Low Confidence Predictions: {result['error_analysis']['low_confidence_count']}
- Average Confidence: {result['error_analysis']['avg_confidence']:.4f}

"""
        
        # Add recommendations
        markdown_content += "\n## Recommendations\n\n"
        for i, rec in enumerate(report['recommendations'], 1):
            markdown_content += f"{i}. {rec}\n\n"
        
        # Save markdown report
        report_file = self.reports_dir / f"benchmark_report_{self.session_id}.md"
        with open(report_file, 'w') as f:
            f.write(markdown_content)
        
        logger.info(f"Markdown report saved: {report_file}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="OCR Model Benchmarking Tool")
    
    # Model arguments
    parser.add_argument("--model", help="Path to single model for benchmarking")
    parser.add_argument("--compare-models", nargs='+', help="Paths to multiple models for comparison")
    
    # Data arguments
    parser.add_argument("--test-data", required=True, 
                       help="Path to test data (JSON file or image directory)")
    
    # Output arguments
    parser.add_argument("--output-dir", default="./benchmark_results",
                       help="Output directory for results")
    
    # Options
    parser.add_argument("--visualize", action="store_true", default=True,
                       help="Generate visualization plots")
    parser.add_argument("--save-results", action="store_true", default=True,
                       help="Save detailed results to files")
    parser.add_argument("--full-report", action="store_true",
                       help="Generate comprehensive report")
    
    return parser.parse_args()


def main():
    """Main benchmarking function"""
    args = parse_args()
    
    if not args.model and not args.compare_models:
        print("Error: Must specify either --model or --compare-models")
        return 1
    
    try:
        # Initialize benchmark tool
        benchmark = OCRBenchmark(args.output_dir)
        
        # Load test data
        logger.info("Loading test data...")
        test_images, test_labels, filenames = benchmark.load_test_data(args.test_data)
        logger.info(f"Loaded {len(test_images)} test samples")
        
        # Run benchmarks
        if args.compare_models:
            # Compare multiple models
            results = benchmark.compare_models(args.compare_models, test_images, test_labels)
        else:
            # Benchmark single model
            result = benchmark.benchmark_model(args.model, test_images, test_labels)
            results = [result]
        
        # Generate visualizations
        if args.visualize:
            for result in results:
                benchmark.generate_visualizations(result)
        
        # Generate comprehensive report
        if args.full_report or len(results) > 1:
            report = benchmark.generate_report(results, save_json=args.save_results)
            
            # Print summary
            print("\n" + "="*50)
            print("BENCHMARK SUMMARY")
            print("="*50)
            for result in results:
                print(f"\n{result.model_name}:")
                print(f"  Accuracy: {result.accuracy:.4f}")
                print(f"  Speed: {result.inference_time_ms:.2f}ms")
                print(f"  Size: {result.model_size_mb:.2f}MB")
            
            print("\nRecommendations:")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        logger.info("Benchmarking completed successfully!")
        
    except Exception as e:
        logger.error(f"Benchmarking failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
