#!/usr/bin/env python3
"""
OCR Benchmark Tool for Smart Meter Reader OCR project.

This script evaluates the performance of the OCR model by measuring accuracy, 
inference time, and resource usage on a set of test images.
"""

import os
import sys
import argparse
import json
import time
import csv
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import psutil
import tensorflow as tf
from tqdm import tqdm
import pandas as pd

# Add project root to path so we can import the OCR module
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='OCR Benchmark Tool')
    
    # Input sources
    parser.add_argument('--test-data', required=True, 
                       help='Path to test data directory (images with labels)')
    parser.add_argument('--model', required=True, 
                       help='Path to TFLite model file')
    
    # Testing parameters
    parser.add_argument('--limit', type=int, default=0, 
                       help='Limit number of test images (0 for all)')
    parser.add_argument('--save-results', action='store_true', 
                       help='Save benchmark results to file')
    parser.add_argument('--output-dir', default='./benchmark_results', 
                       help='Output directory for results')
    parser.add_argument('--visualize', action='store_true', 
                       help='Visualize results (show confusion matrix etc.)')
    parser.add_argument('--warm-up', type=int, default=3, 
                       help='Number of warm-up runs before timing')
    
    # Advanced options
    parser.add_argument('--preprocess', choices=['none', 'basic', 'advanced'], default='basic', 
                       help='Preprocessing level')
    parser.add_argument('--num-threads', type=int, default=1, 
                       help='Number of threads for inference')
    parser.add_argument('--detailed-metrics', action='store_true', 
                       help='Report detailed metrics for each digit')
    parser.add_argument('--per-device-metrics', action='store_true',
                       help='Report metrics for each device/meter type')
    
    return parser.parse_args()

def load_test_data(data_path, limit=0):
    """
    Load test images and their labels.
    
    Args:
        data_path: Path to test data directory
        limit: Maximum number of images to load (0 for all)
        
    Returns:
        Tuple of (image_paths, labels)
    """
    if not os.path.isdir(data_path):
        raise ValueError(f"Test data path does not exist: {data_path}")
    
    # Try to find labels.csv file
    labels_file = os.path.join(data_path, "labels.csv")
    if os.path.isfile(labels_file):
        print(f"Found labels file: {labels_file}")
        return load_from_csv(labels_file, data_path, limit)
    
    # If no CSV, try to find image files with corresponding JSON metadata
    print("No labels.csv found, looking for image files with JSON metadata...")
    return load_from_files(data_path, limit)

def load_from_csv(csv_file, data_path, limit=0):
    """Load test data from CSV file."""
    image_paths = []
    labels = []
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'filename' in row and 'label' in row:
                image_path = os.path.join(data_path, row['filename'])
                if os.path.isfile(image_path):
                    image_paths.append(image_path)
                    labels.append(row['label'])
                    
                    if limit > 0 and len(image_paths) >= limit:
                        break
    
    print(f"Loaded {len(image_paths)} images from CSV")
    return image_paths, labels

def load_from_files(data_path, limit=0):
    """Load test data by searching for image files with JSON metadata."""
    image_paths = []
    labels = []
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(data_path, f"*{ext}")))
        image_files.extend(glob.glob(os.path.join(data_path, f"*{ext.upper()}")))
    
    # Sort for consistency
    image_files.sort()
    
    # Limit if needed
    if limit > 0:
        image_files = image_files[:limit]
    
    # Load labels from JSON files
    for image_file in image_files:
        json_file = os.path.splitext(image_file)[0] + ".json"
        if os.path.isfile(json_file):
            try:
                with open(json_file, 'r') as f:
                    metadata = json.load(f)
                
                if "label" in metadata:
                    image_paths.append(image_file)
                    labels.append(metadata["label"])
            except Exception as e:
                print(f"Warning: Could not read metadata from {json_file}: {e}")
    
    print(f"Loaded {len(image_paths)} images with metadata from files")
    return image_paths, labels

def load_tflite_model(model_path):
    """
    Load TFLite model for inference.
    
    Args:
        model_path: Path to TFLite model file
        
    Returns:
        TFLite interpreter
    """
    if not os.path.isfile(model_path):
        raise ValueError(f"Model file does not exist: {model_path}")
    
    # Load TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Model loaded: {model_path}")
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Output shape: {output_details[0]['shape']}")
    
    return interpreter, input_details, output_details

def preprocess_image(image_path, input_shape, level='basic'):
    """
    Preprocess image for inference.
    
    Args:
        image_path: Path to image file
        input_shape: Shape of model input (from input_details)
        level: Preprocessing level ('none', 'basic', or 'advanced')
        
    Returns:
        Preprocessed image as numpy array
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if level == 'none':
        # Minimal preprocessing - just resize
        resized = cv2.resize(gray, (input_shape[1], input_shape[2]))
        
    elif level == 'basic':
        # Basic preprocessing
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # Resize
        resized = cv2.resize(thresh, (input_shape[1], input_shape[2]))
        
    elif level == 'advanced':
        # Advanced preprocessing
        # Apply bilateral filter to remove noise while preserving edges
        filtered = cv2.bilateralFilter(gray, 5, 75, 75)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # If contours found, focus on the largest one
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Extract ROI (region of interest)
            roi = gray[y:y+h, x:x+w]
            
            # Apply Otsu's thresholding
            _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Resize
            resized = cv2.resize(binary, (input_shape[1], input_shape[2]))
        else:
            # Fallback to basic if no contours found
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 11, 2)
            resized = cv2.resize(thresh, (input_shape[1], input_shape[2]))
    
    # Normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0
    
    # Add batch dimension
    batched = np.expand_dims(normalized, axis=0)
    
    # Add channel dimension if needed
    if len(input_shape) == 4 and input_shape[3] == 1:
        batched = np.expand_dims(batched, axis=-1)
    
    return batched

def run_inference(interpreter, input_details, output_details, preprocessed_image):
    """
    Run inference on preprocessed image.
    
    Args:
        interpreter: TFLite interpreter
        input_details: Model input details
        output_details: Model output details
        preprocessed_image: Preprocessed image as numpy array
        
    Returns:
        Inference result and execution time
    """
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], preprocessed_image)
    
    # Measure inference time
    start_time = time.time()
    
    # Run inference
    interpreter.invoke()
    
    # Get execution time
    execution_time = time.time() - start_time
    
    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    return output_data, execution_time

def postprocess_output(output_data, metadata=None):
    """
    Postprocess model output to get the predicted text.
    
    Args:
        output_data: Raw model output
        metadata: Optional model metadata for interpretation
        
    Returns:
        Predicted text (string)
    """
    # This is a placeholder implementation
    # The actual implementation would depend on the model's output format
    
    # For classification model with one digit per position
    if len(output_data.shape) == 2:
        # Assuming output is [batch_size, num_classes]
        predicted_class = np.argmax(output_data[0])
        return str(predicted_class)
    
    # For multi-digit classification
    elif len(output_data.shape) == 3:
        # Assuming output is [batch_size, num_digits, num_classes]
        predicted_digits = []
        for digit_probs in output_data[0]:
            predicted_digit = np.argmax(digit_probs)
            predicted_digits.append(str(predicted_digit))
        
        return ''.join(predicted_digits)
    
    # For sequence models (like CTC)
    elif len(output_data.shape) == 4:
        # This would require CTC decoding - simplified here
        # Assuming output is [batch_size, time_steps, num_classes]
        predicted_chars = []
        for time_step in range(output_data.shape[1]):
            char_idx = np.argmax(output_data[0, time_step, 0])
            if char_idx > 0:  # Skip blank (usually 0)
                predicted_chars.append(str(char_idx - 1))  # Adjust index
        
        return ''.join(predicted_chars)
    
    # Fallback for unknown format
    else:
        return str(output_data)

def calculate_metrics(true_labels, predicted_labels, detailed=False):
    """
    Calculate accuracy and other metrics.
    
    Args:
        true_labels: List of ground truth labels
        predicted_labels: List of predicted labels
        detailed: Whether to calculate per-digit metrics
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Calculate overall accuracy
    correct = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred)
    metrics['accuracy'] = correct / len(true_labels) if len(true_labels) > 0 else 0
    
    # Calculate total character error rate (CER)
    total_chars = sum(len(label) for label in true_labels)
    char_errors = sum(levenshtein_distance(true, pred) for true, pred in zip(true_labels, predicted_labels))
    metrics['character_error_rate'] = char_errors / total_chars if total_chars > 0 else 0
    
    # Calculate per-digit accuracy if requested
    if detailed:
        # Get maximum label length
        max_len = max(len(label) for label in true_labels + predicted_labels)
        
        # Initialize per-digit metrics
        per_digit_correct = [0] * max_len
        per_digit_total = [0] * max_len
        
        for true, pred in zip(true_labels, predicted_labels):
            # Pad shorter label with spaces
            true_padded = true.ljust(max_len)
            pred_padded = pred.ljust(max_len)
            
            for i in range(max_len):
                if i < len(true):
                    per_digit_total[i] += 1
                    if i < len(pred) and true[i] == pred[i]:
                        per_digit_correct[i] += 1
        
        # Calculate per-digit accuracy
        metrics['per_digit_accuracy'] = []
        for i in range(max_len):
            if per_digit_total[i] > 0:
                accuracy = per_digit_correct[i] / per_digit_total[i]
            else:
                accuracy = 0
            metrics['per_digit_accuracy'].append(accuracy)
    
    return metrics

def levenshtein_distance(s1, s2):
    """
    Calculate Levenshtein distance between two strings.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Levenshtein distance
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Calculate insertions, deletions, and substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            
            # Get minimum
            current_row.append(min(insertions, deletions, substitutions))
        
        previous_row = current_row
    
    return previous_row[-1]

def measure_cpu_memory():
    """
    Measure CPU and memory usage.
    
    Returns:
        Tuple of (cpu_percent, memory_mb)
    """
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory_info = psutil.Process(os.getpid()).memory_info()
    memory_mb = memory_info.rss / 1024 / 1024  # Convert bytes to MB
    
    return cpu_percent, memory_mb

def visualize_results(true_labels, predicted_labels, execution_times, metrics, output_dir):
    """
    Visualize benchmark results.
    
    Args:
        true_labels: List of ground truth labels
        predicted_labels: List of predicted labels
        execution_times: List of execution times
        metrics: Dictionary of metrics
        output_dir: Output directory for saving plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot execution time histogram
    plt.figure(figsize=(10, 6))
    plt.hist(execution_times, bins=20)
    plt.title('Inference Time Distribution')
    plt.xlabel('Execution Time (s)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'execution_time_histogram.png'))
    plt.close()
    
    # Plot per-digit accuracy if available
    if 'per_digit_accuracy' in metrics:
        plt.figure(figsize=(10, 6))
        positions = list(range(1, len(metrics['per_digit_accuracy']) + 1))
        plt.bar(positions, metrics['per_digit_accuracy'])
        plt.title('Per-Digit Accuracy')
        plt.xlabel('Digit Position')
        plt.ylabel('Accuracy')
        plt.xticks(positions)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'per_digit_accuracy.png'))
        plt.close()
    
    # Create confusion matrix for single digits
    all_true_digits = ''.join(true_labels)
    all_pred_digits = ''.join(predicted_labels)
    
    # Truncate or pad prediction to match true length
    all_pred_digits = all_pred_digits[:len(all_true_digits)].ljust(len(all_true_digits))
    
    # Get unique digits
    unique_digits = sorted(set(all_true_digits + all_pred_digits))
    
    # Initialize confusion matrix
    confusion_matrix = np.zeros((len(unique_digits), len(unique_digits)), dtype=int)
    
    # Fill confusion matrix
    for true_digit, pred_digit in zip(all_true_digits, all_pred_digits):
        true_idx = unique_digits.index(true_digit)
        pred_idx = unique_digits.index(pred_digit)
        confusion_matrix[true_idx][pred_idx] += 1
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(unique_digits))
    plt.xticks(tick_marks, unique_digits)
    plt.yticks(tick_marks, unique_digits)
    plt.xlabel('Predicted Digit')
    plt.ylabel('True Digit')
    
    # Add text annotations
    thresh = confusion_matrix.max() / 2
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            plt.text(j, i, confusion_matrix[i, j],
                   horizontalalignment="center",
                   color="white" if confusion_matrix[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def save_results_to_file(image_paths, true_labels, predicted_labels, execution_times, metrics, output_dir):
    """
    Save benchmark results to file.
    
    Args:
        image_paths: List of image paths
        true_labels: List of ground truth labels
        predicted_labels: List of predicted labels
        execution_times: List of execution times
        metrics: Dictionary of metrics
        output_dir: Output directory for saving results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results as CSV
    results_file = os.path.join(output_dir, 'benchmark_results.csv')
    with open(results_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'true_label', 'predicted_label', 'correct', 'execution_time'])
        
        for image_path, true, pred, time in zip(image_paths, true_labels, predicted_labels, execution_times):
            correct = true == pred
            writer.writerow([os.path.basename(image_path), true, pred, correct, time])
    
    # Save summary metrics as JSON
    metrics_file = os.path.join(output_dir, 'benchmark_metrics.json')
    with open(metrics_file, 'w') as f:
        # Add some additional summary stats
        metrics['avg_execution_time'] = sum(execution_times) / len(execution_times)
        metrics['min_execution_time'] = min(execution_times)
        metrics['max_execution_time'] = max(execution_times)
        
        json.dump(metrics, f, indent=4)
    
    print(f"Results saved to {output_dir}")

def main():
    """Main entry point of the script."""
    args = parse_args()
    
    # Load test data
    image_paths, true_labels = load_test_data(args.test_data, args.limit)
    
    if not image_paths:
        print("Error: No test data found")
        return
    
    # Load model
    interpreter, input_details, output_details = load_tflite_model(args.model)
    
    # Set number of threads
    interpreter.set_num_threads(args.num_threads)
    
    # Perform warm-up runs to eliminate initialization overhead
    if args.warm_up > 0:
        print(f"Performing {args.warm_up} warm-up runs...")
        for _ in range(args.warm_up):
            test_image = preprocess_image(image_paths[0], input_details[0]['shape'], args.preprocess)
            interpreter.set_tensor(input_details[0]['index'], test_image)
            interpreter.invoke()
    
    # Run benchmark
    print(f"Running benchmark on {len(image_paths)} images...")
    predicted_labels = []
    execution_times = []
    cpu_usages = []
    memory_usages = []
    
    for i, (image_path, true_label) in enumerate(tqdm(zip(image_paths, true_labels), total=len(image_paths))):
        try:
            # Preprocess image
            preprocessed_image = preprocess_image(image_path, input_details[0]['shape'], args.preprocess)
            
            # Measure CPU and memory before inference
            cpu_before, memory_before = measure_cpu_memory()
            
            # Run inference
            output_data, execution_time = run_inference(
                interpreter, input_details, output_details, preprocessed_image)
            
            # Measure CPU and memory after inference
            cpu_after, memory_after = measure_cpu_memory()
            
            # Postprocess output
            predicted_label = postprocess_output(output_data)
            
            # Record results
            predicted_labels.append(predicted_label)
            execution_times.append(execution_time)
            cpu_usages.append(cpu_after - cpu_before)
            memory_usages.append(memory_after - memory_before)
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            # Add placeholders for failed inference
            predicted_labels.append("")
            execution_times.append(0)
            cpu_usages.append(0)
            memory_usages.append(0)
    
    # Calculate metrics
    metrics = calculate_metrics(true_labels, predicted_labels, args.detailed_metrics)
    
    # Add resource usage metrics
    metrics['avg_cpu_usage'] = sum(cpu_usages) / len(cpu_usages) if cpu_usages else 0
    metrics['avg_memory_usage_mb'] = sum(memory_usages) / len(memory_usages) if memory_usages else 0
    
    # Print summary
    print("\nBenchmark Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Character Error Rate: {metrics['character_error_rate']:.4f}")
    print(f"Average Inference Time: {sum(execution_times) / len(execution_times):.4f} seconds")
    print(f"Average CPU Usage: {metrics['avg_cpu_usage']:.2f}%")
    print(f"Average Memory Usage: {metrics['avg_memory_usage_mb']:.2f} MB")
    
    # Visualize results if requested
    if args.visualize:
        print("\nGenerating visualizations...")
        visualize_results(true_labels, predicted_labels, execution_times, metrics, args.output_dir)
    
    # Save results if requested
    if args.save_results:
        print("\nSaving results...")
        save_results_to_file(image_paths, true_labels, predicted_labels, execution_times, metrics, args.output_dir)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBenchmark aborted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
