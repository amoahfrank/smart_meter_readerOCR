#!/usr/bin/env python3
"""
Data Collection Tool for Smart Meter Reader OCR project.

This script helps automate the collection of meter readings from real-world meters 
or simulated data, organizing and labeling them for training the OCR model.
"""

import os
import sys
import argparse
import cv2
import numpy as np
import json
import time
import datetime
import serial
from serial.tools import list_ports
import shutil
import csv
from PIL import Image
import random
import glob

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Data collection for Smart Meter Reader OCR')
    
    # Device connection options
    parser.add_argument('--device', choices=['esp32', 'webcam', 'simulator', 'import'], 
                        default='webcam', help='Data source')
    parser.add_argument('--port', help='Serial port for ESP32 device')
    parser.add_argument('--baud', type=int, default=115200, help='Baud rate for ESP32 (default: 115200)')
    parser.add_argument('--camera-id', type=int, default=0, help='Camera ID for webcam (default: 0)')
    
    # Data collection parameters
    parser.add_argument('--output-dir', default='./collected_data', help='Output directory')
    parser.add_argument('--count', type=int, default=10, help='Number of images to collect')
    parser.add_argument('--interval', type=float, default=1.0, help='Interval between captures (seconds)')
    parser.add_argument('--prefix', default='meter', help='Filename prefix')
    parser.add_argument('--auto-label', action='store_true', help='Try to auto-label values (only for simulator)')
    parser.add_argument('--import-path', help='Path to import existing images from')
    
    # Display options
    parser.add_argument('--preview', action='store_true', help='Show preview window')
    parser.add_argument('--resolution', default='640x480', 
                        help='Capture resolution as WIDTHxHEIGHT (default: 640x480)')
    
    return parser.parse_args()

def find_esp32_port():
    """Attempt to automatically find the ESP32 serial port."""
    ports = list_ports.comports()
    for port in ports:
        # Look for common ESP32 USB-to-UART bridge chips
        if "CP210X" in port.description.upper() or "CH340" in port.description.upper() or "FTDI" in port.description.upper():
            print(f"Found potential ESP32 device: {port.device} - {port.description}")
            return port.device
    return None

def connect_to_device(port, baud_rate):
    """Establish serial connection to the ESP32 device."""
    try:
        ser = serial.Serial(port, baud_rate, timeout=2)
        print(f"Connected to {port} at {baud_rate} baud")
        # Wait for ESP32 to reset
        time.sleep(2)
        ser.reset_input_buffer()
        return ser
    except serial.SerialException as e:
        print(f"Error connecting to device: {e}")
        return None

def send_command(ser, command):
    """Send command to ESP32 and read response."""
    if not ser:
        print("Error: No serial connection")
        return None
    
    try:
        ser.write((command + '\n').encode())
        time.sleep(0.5)  # Give ESP32 time to process
        
        response = ""
        while ser.in_waiting:
            response += ser.readline().decode('utf-8', errors='ignore')
        
        return response
    except Exception as e:
        print(f"Error communicating with device: {e}")
        return None

def setup_webcam(camera_id, resolution):
    """Initialize webcam for data collection."""
    width, height = map(int, resolution.split('x'))
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera with ID {camera_id}")
        return None
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    print(f"Webcam initialized at resolution {width}x{height}")
    return cap

def capture_from_esp32(ser):
    """Capture image from ESP32 device."""
    response = send_command(ser, "CAPTURE")
    if not response or "ERROR" in response:
        print("Failed to capture image from ESP32")
        return None
    
    # Parse image data - This is a placeholder
    # In a real implementation, you would need a protocol to receive
    # binary image data from the ESP32
    print("Image captured from ESP32 - binary data transfer required")
    
    # Return a dummy image for this example
    return np.zeros((480, 640, 3), dtype=np.uint8)

def capture_from_webcam(cap):
    """Capture image from webcam."""
    if not cap or not cap.isOpened():
        print("Error: Webcam not initialized")
        return None
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image from webcam")
        return None
    
    return frame

def simulate_meter_reading():
    """Simulate a meter reading for testing."""
    try:
        # Try to import the simulator if it's available
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'simulator'))
        from meter_simulator import MeterSimulator
        
        # Create a simulator instance with random settings
        meter_type = random.choice(['lcd_digital', 'led_digital', 'rotary_dial'])
        digit_count = random.randint(4, 6)
        noise_level = random.uniform(0, 0.2)
        blur_level = random.uniform(0, 0.3)
        light_condition = random.choice(['normal', 'dim', 'bright', 'glare'])
        
        settings = {
            "meter_type": meter_type,
            "digit_count": digit_count,
            "noise_level": noise_level,
            "blur_level": blur_level,
            "light_condition": light_condition,
            "width": 640,
            "height": 480
        }
        
        simulator = MeterSimulator(settings)
        
        # Generate a random value
        value = random.randint(0, 10**digit_count - 1)
        
        # Generate the image
        image, formatted_value = simulator.generate_image(value)
        
        # Convert PIL image to OpenCV format
        img_array = np.array(image)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        return img_array, formatted_value, settings
    
    except ImportError:
        print("Meter simulator module not found, using placeholder image")
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, "SIMULATED METER", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return img, "12345", {"meter_type": "simulated"}

def save_image_with_metadata(image, label, output_dir, prefix, metadata=None):
    """Save captured image with metadata."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{prefix}_{timestamp}.jpg"
    filepath = os.path.join(output_dir, filename)
    
    # Save image
    cv2.imwrite(filepath, image)
    
    # Save metadata
    metadata_file = f"{os.path.splitext(filepath)[0]}.json"
    if metadata is None:
        metadata = {}
    
    metadata["timestamp"] = timestamp
    metadata["label"] = label
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    # Update labels CSV
    labels_file = os.path.join(output_dir, "labels.csv")
    file_exists = os.path.isfile(labels_file)
    
    with open(labels_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["filename", "label", "timestamp"])
        writer.writerow([filename, label, timestamp])
    
    print(f"Image saved to {filepath} with label: {label}")
    return filepath

def import_existing_images(import_path, output_dir, prefix):
    """Import existing images from a directory."""
    if not os.path.isdir(import_path):
        print(f"Error: Import path {import_path} is not a directory")
        return 0
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Look for images
    image_files = []
    for ext in ['jpg', 'jpeg', 'png', 'bmp']:
        image_files.extend(glob.glob(os.path.join(import_path, f"*.{ext}")))
        image_files.extend(glob.glob(os.path.join(import_path, f"*.{ext.upper()}")))
    
    if not image_files:
        print(f"No image files found in {import_path}")
        return 0
    
    print(f"Found {len(image_files)} images to import")
    imported_count = 0
    
    for image_file in image_files:
        try:
            # Generate new filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{prefix}_{timestamp}.jpg"
            filepath = os.path.join(output_dir, filename)
            
            # Copy image to output directory
            shutil.copy2(image_file, filepath)
            
            # Try to find a corresponding JSON file with metadata
            json_file = os.path.splitext(image_file)[0] + ".json"
            metadata = None
            label = "unknown"
            
            if os.path.isfile(json_file):
                try:
                    with open(json_file, 'r') as f:
                        metadata = json.load(f)
                    if "label" in metadata or "value" in metadata:
                        label = metadata.get("label", metadata.get("value", "unknown"))
                    shutil.copy2(json_file, f"{os.path.splitext(filepath)[0]}.json")
                except Exception as e:
                    print(f"Warning: Could not read metadata from {json_file}: {e}")
            
            # If no metadata file or no label, ask for manual label
            if label == "unknown":
                img = cv2.imread(filepath)
                if img is not None:
                    # Display image and ask for label
                    cv2.imshow("Image for labeling", img)
                    cv2.waitKey(100)  # Small delay to ensure window is shown
                    label = input(f"Enter label for {os.path.basename(image_file)}: ")
                    cv2.destroyAllWindows()
            
            # Save metadata
            metadata_file = f"{os.path.splitext(filepath)[0]}.json"
            if metadata is None:
                metadata = {}
            
            metadata["timestamp"] = timestamp
            metadata["label"] = label
            metadata["original_file"] = os.path.basename(image_file)
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            # Update labels CSV
            labels_file = os.path.join(output_dir, "labels.csv")
            file_exists = os.path.isfile(labels_file)
            
            with open(labels_file, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["filename", "label", "timestamp", "original_file"])
                writer.writerow([filename, label, timestamp, os.path.basename(image_file)])
            
            imported_count += 1
            print(f"Imported {image_file} as {filepath} with label: {label}")
            
        except Exception as e:
            print(f"Error importing {image_file}: {e}")
    
    print(f"Successfully imported {imported_count} images")
    return imported_count

def main():
    """Main entry point of the script."""
    args = parse_args()
    
    # Check if output directory exists, create if not
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Handle import mode
    if args.device == 'import':
        if not args.import_path:
            print("Error: --import-path must be specified for import mode")
            return
        
        imported_count = import_existing_images(args.import_path, args.output_dir, args.prefix)
        print(f"Import completed. {imported_count} images processed.")
        return
    
    # Initialize data source
    data_source = None
    
    if args.device == 'esp32':
        # Find port if not specified
        port = args.port
        if not port:
            port = find_esp32_port()
            if not port:
                print("Error: Could not find ESP32 device. Please specify port with --port")
                return
        
        # Connect to ESP32
        data_source = connect_to_device(port, args.baud)
        if not data_source:
            return
    
    elif args.device == 'webcam':
        # Initialize webcam
        data_source = setup_webcam(args.camera_id, args.resolution)
        if not data_source:
            return
    
    elif args.device == 'simulator':
        # No initialization needed for simulator
        pass
    
    # Start data collection
    print(f"Starting data collection. Will capture {args.count} images.")
    
    captured_count = 0
    try:
        while captured_count < args.count:
            # Capture image
            if args.device == 'esp32':
                image = capture_from_esp32(data_source)
                label = input("Enter meter reading value: ")
                metadata = {"device": "esp32"}
            
            elif args.device == 'webcam':
                image = capture_from_webcam(data_source)
                
                # Show preview if requested
                if args.preview and image is not None:
                    cv2.imshow("Preview", image)
                    cv2.waitKey(1)
                
                label = input("Enter meter reading value: ")
                metadata = {"device": "webcam"}
            
            elif args.device == 'simulator':
                image, label, metadata = simulate_meter_reading()
                metadata["device"] = "simulator"
                
                # Show preview if requested
                if args.preview and image is not None:
                    cv2.imshow("Preview", image)
                    cv2.waitKey(1)
                
                # Override label if auto-label is not enabled
                if not args.auto_label:
                    print(f"Simulated value: {label}")
                    user_label = input("Enter meter reading value (leave blank to use simulated value): ")
                    if user_label:
                        label = user_label
            
            # Save image if captured successfully
            if image is not None:
                filepath = save_image_with_metadata(image, label, args.output_dir, args.prefix, metadata)
                captured_count += 1
                print(f"Captured {captured_count}/{args.count}")
            
            # Wait for interval
            if captured_count < args.count:
                time.sleep(args.interval)
    
    except KeyboardInterrupt:
        print("\nData collection aborted by user")
    
    finally:
        # Clean up
        if args.device == 'webcam' and data_source:
            data_source.release()
        
        if args.preview:
            cv2.destroyAllWindows()
        
        if args.device == 'esp32' and data_source:
            data_source.close()
    
    print(f"Data collection completed. {captured_count} images captured.")

if __name__ == "__main__":
    main()
