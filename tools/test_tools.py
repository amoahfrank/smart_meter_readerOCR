#!/usr/bin/env python3
"""
Test script for Smart Meter Reader OCR tools.

This script runs a basic integration test to ensure all tools are functioning correctly
and can work together in a typical development workflow.
"""

import os
import sys
import subprocess
import argparse
import shutil
import time
import glob

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test Smart Meter Reader OCR tools')
    parser.add_argument('--output-dir', default='./test_output', help='Output directory for test results')
    parser.add_argument('--clean', action='store_true', help='Clean output directory before starting')
    parser.add_argument('--quick', action='store_true', help='Run quick test with minimal data')
    return parser.parse_args()

def run_command(cmd, description, exit_on_error=True):
    """Run a command and print output."""
    print(f"\n{'='*80}")
    print(f"Running {description}...")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'-'*80}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print(f"Exit code: {result.returncode}")
    
    if result.stdout:
        print("\nStandard output:")
        print(result.stdout)
    
    if result.stderr:
        print("\nStandard error:")
        print(result.stderr)
    
    if exit_on_error and result.returncode != 0:
        print(f"Error running {description}, exiting.")
        sys.exit(1)
    
    return result.returncode == 0

def check_tool_exists(tool_path):
    """Check if a tool script exists."""
    if not os.path.isfile(tool_path):
        print(f"Tool not found: {tool_path}")
        return False
    return True

def main():
    """Main entry point of the script."""
    args = parse_args()
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Prepare output directory
    output_dir = os.path.abspath(args.output_dir)
    
    if args.clean and os.path.exists(output_dir):
        print(f"Cleaning output directory: {output_dir}")
        shutil.rmtree(output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Define tool paths
    simulator_path = os.path.join(script_dir, 'simulator', 'meter_simulator.py')
    data_collection_path = os.path.join(script_dir, 'data_collection', 'collect_meter_data.py')
    model_converter_path = os.path.join(script_dir, 'model_converter', 'convert_model.py')
    benchmark_path = os.path.join(script_dir, 'benchmark', 'ocr_benchmark.py')
    
    # Check if tools exist
    for tool_path in [simulator_path, data_collection_path, model_converter_path, benchmark_path]:
        if not check_tool_exists(tool_path):
            print("Some tools are missing, please make sure all tools are installed correctly.")
            sys.exit(1)
    
    # Create paths for test workflow
    simulated_data_dir = os.path.join(output_dir, 'simulated_data')
    collected_data_dir = os.path.join(output_dir, 'collected_data')
    model_dir = os.path.join(output_dir, 'model')
    benchmark_dir = os.path.join(output_dir, 'benchmark')
    
    os.makedirs(simulated_data_dir, exist_ok=True)
    os.makedirs(collected_data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(benchmark_dir, exist_ok=True)
    
    # Test workflow
    
    # 1. Generate simulated meter data
    batch_size = 2 if args.quick else 10
    
    for meter_type in ['lcd_digital', 'led_digital', 'rotary_dial']:
        cmd = [
            sys.executable, simulator_path,
            '--meter-type', meter_type,
            '--batch-size', str(batch_size),
            '--output-dir', simulated_data_dir,
            '--digit-count', '5'
        ]
        
        run_command(cmd, f"meter simulation ({meter_type})")
    
    # 2. Import simulated data using data collection tool
    cmd = [
        sys.executable, data_collection_path,
        '--device', 'import',
        '--import-path', simulated_data_dir,
        '--output-dir', collected_data_dir
    ]
    
    run_command(cmd, "data collection (import)")
    
    # 3. Create a dummy model file for testing model converter
    # (In a real workflow, you would have an actual model trained on your data)
    dummy_model_path = os.path.join(model_dir, 'dummy_model.h5')
    tflite_model_path = os.path.join(model_dir, 'model.tflite')
    
    try:
        import tensorflow as tf
        from tensorflow import keras
        import numpy as np
        
        # Create a very simple model for testing
        inputs = keras.Input(shape=(48, 48, 1))
        x = keras.layers.Conv2D(8, 3, activation='relu')(inputs)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Flatten()(x)
        outputs = keras.layers.Dense(10, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        
        # Save the model
        model.save(dummy_model_path)
        print(f"Created dummy model: {dummy_model_path}")
        
        # 4. Convert model to TFLite
        cmd = [
            sys.executable, model_converter_path,
            '--input_model', dummy_model_path,
            '--output_model', tflite_model_path,
            '--quantize'
        ]
        
        run_command(cmd, "model conversion")
        
        # 5. Run benchmark on test data
        # Since we're using a dummy model, we don't expect accurate results
        # This is just to test that the benchmark tool runs
        cmd = [
            sys.executable, benchmark_path,
            '--test-data', collected_data_dir,
            '--model', tflite_model_path,
            '--output-dir', benchmark_dir,
            '--save-results',
            '--limit', '2'  # Just test with a few images
        ]
        
        run_command(cmd, "OCR benchmark", exit_on_error=False)
        
    except ImportError:
        print("TensorFlow not installed, skipping model conversion and benchmark steps.")
    
    # Print summary
    print("\n" + "="*80)
    print("Tool tests completed!")
    print(f"Output directory: {output_dir}")
    
    # Check what was generated
    print("\nGenerated files:")
    for root, dirs, files in os.walk(output_dir):
        level = root.replace(output_dir, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = ' ' * 4 * (level + 1)
        
        # Only show a few files in each directory
        file_count = len(files)
        files_to_show = files[:3]
        
        for f in files_to_show:
            print(f"{sub_indent}{f}")
        
        if file_count > len(files_to_show):
            print(f"{sub_indent}... ({file_count - len(files_to_show)} more files)")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTest aborted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
