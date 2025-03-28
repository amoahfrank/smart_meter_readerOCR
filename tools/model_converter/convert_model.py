#!/usr/bin/env python3
"""
Convert TensorFlow models to TensorFlow Lite for use in the Smart Meter Reader OCR project.

This script handles conversion of trained digit recognition models to TFLite format
optimized for microcontroller deployment.
"""

import os
import argparse
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Convert TensorFlow model to TFLite')
    parser.add_argument('--input_model', required=True, 
                        help='Path to input Keras model (.h5)')
    parser.add_argument('--output_model', required=True, 
                        help='Path to output TFLite model (.tflite)')
    parser.add_argument('--quantize', action='store_true', 
                        help='Apply int8 quantization to optimize model size')
    parser.add_argument('--representative_dataset', 
                        help='Path to representative dataset for quantization calibration')
    parser.add_argument('--input_shape', default='48,48,1', 
                        help='Input shape as comma-separated values (default: 48,48,1)')
    return parser.parse_args()

def prepare_representative_dataset(dataset_path, input_shape):
    """Load representative dataset for quantization."""
    if not dataset_path or not os.path.exists(dataset_path):
        print("Warning: No valid representative dataset provided, using random data")
        # Create a function that generates random input data
        input_shape_list = [int(dim) for dim in input_shape.split(',')]
        
        def representative_dataset():
            for _ in range(100):
                data = np.random.rand(1, *input_shape_list).astype(np.float32)
                yield [data]
                
        return representative_dataset
    
    # Load actual representative dataset
    # This is a placeholder - implement actual loading based on your dataset format
    print(f"Loading representative dataset from {dataset_path}")
    
    # Simple CSV implementation - modify as needed
    images = np.loadtxt(dataset_path, delimiter=',').astype(np.float32)
    input_shape_list = [int(dim) for dim in input_shape.split(',')]
    images = images.reshape(-1, *input_shape_list)
    
    def representative_dataset():
        for image in images:
            yield [np.expand_dims(image, axis=0)]
    
    return representative_dataset

def convert_to_tflite(model_path, output_path, quantize=False, representative_dataset=None, input_shape=None):
    """Convert Keras model to TFLite format."""
    print(f"Loading model from {model_path}")
    model = load_model(model_path)
    model.summary()
    
    # Create TFLite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Set optimization flags
    if quantize:
        print("Applying int8 quantization")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        if representative_dataset:
            print("Using representative dataset for quantization")
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
    
    # Convert model
    tflite_model = converter.convert()
    
    # Save model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Model converted and saved to {output_path}")
    print(f"Model size: {os.path.getsize(output_path) / 1024:.2f} KB")

def main():
    """Main entry point of the script."""
    args = parse_args()
    
    # Validate paths
    if not os.path.exists(args.input_model):
        raise FileNotFoundError(f"Input model file not found: {args.input_model}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output_model)), exist_ok=True)
    
    # Prepare representative dataset if quantization is requested
    representative_dataset = None
    if args.quantize and args.representative_dataset:
        representative_dataset = prepare_representative_dataset(
            args.representative_dataset, 
            args.input_shape
        )
    
    # Convert model
    convert_to_tflite(
        args.input_model,
        args.output_model,
        quantize=args.quantize,
        representative_dataset=representative_dataset,
        input_shape=args.input_shape
    )
    
    print("Conversion completed successfully!")

if __name__ == "__main__":
    main()
