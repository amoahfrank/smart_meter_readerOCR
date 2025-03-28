#!/usr/bin/env python3
"""
Camera calibration tool for the Smart Meter Reader OCR project.

This script helps calibrate the camera module of the Smart Meter Reader device,
determining optimal settings for meter reading accuracy.
"""

import os
import sys
import argparse
import cv2
import numpy as np
import json
import time
import serial
from serial.tools import list_ports

# Default calibration settings
DEFAULT_SETTINGS = {
    "resolution": "VGA",  # VGA (640x480)
    "contrast": 0,
    "brightness": 0,
    "saturation": 0,
    "sharpness": 0,
    "special_effect": 0,
    "wb_mode": 0,
    "awb_gain": 1,
    "aec_value": 300,
    "aec2": 0,
    "ae_level": 0,
    "agc_gain": 0,
    "bpc": 0,
    "wpc": 1,
    "raw_gma": 1,
    "lens_correction": 1,
    "binning": 0
}

# Resolution options for ESP32-CAM with OV2640
RESOLUTION_MAP = {
    "QVGA": (320, 240),
    "VGA": (640, 480),
    "SVGA": (800, 600),
    "XGA": (1024, 768),
    "SXGA": (1280, 1024),
    "UXGA": (1600, 1200)
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Camera calibration for Smart Meter Reader')
    parser.add_argument('--port', help='Serial port of the device')
    parser.add_argument('--baud', type=int, default=115200, help='Baud rate (default: 115200)')
    parser.add_argument('--output', default='camera_calibration.json', 
                        help='Output file for calibration settings (default: camera_calibration.json)')
    parser.add_argument('--reference', help='Reference image for comparison')
    parser.add_argument('--auto', action='store_true', help='Run automated calibration procedure')
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

def capture_image(ser):
    """Request and receive camera image from ESP32."""
    response = send_command(ser, "CAPTURE")
    if not response or "ERROR" in response:
        print("Failed to capture image")
        return None
    
    # Parse image data - This is a placeholder for actual implementation
    # In a real setup, this would handle JPEG data from the ESP32
    # For this sample script, we're simulating the process
    
    # Simulated: In reality, you would implement a protocol for receiving
    # JPEG image data from ESP32 and decoding it
    print("Image captured - implementing binary data transfer would be needed in real application")
    
    # Return a blank image for demo purposes
    return np.zeros((480, 640, 3), dtype=np.uint8)

def analyze_image_quality(image, reference=None):
    """Analyze image quality metrics relevant for OCR."""
    if image is None:
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Calculate clarity metrics
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Calculate histogram to check exposure
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_normalized = hist / (gray.shape[0] * gray.shape[1])
    
    # Check if image is too dark or too bright
    dark_pixels_ratio = sum(hist_normalized[:50]) / sum(hist_normalized)
    bright_pixels_ratio = sum(hist_normalized[200:]) / sum(hist_normalized)
    
    # Compare with reference image if provided
    ref_similarity = None
    if reference is not None:
        # Convert reference to grayscale if needed
        ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY) if len(reference.shape) == 3 else reference
        
        # Resize reference to match image dimensions
        ref_gray = cv2.resize(ref_gray, (gray.shape[1], gray.shape[0]))
        
        # Calculate structural similarity index
        try:
            from skimage.metrics import structural_similarity as ssim
            ref_similarity, _ = ssim(gray, ref_gray, full=True)
        except ImportError:
            print("scikit-image not available, skipping reference comparison")
    
    return {
        "clarity": laplacian_var,
        "dark_ratio": float(dark_pixels_ratio),
        "bright_ratio": float(bright_pixels_ratio),
        "reference_similarity": ref_similarity
    }

def auto_calibrate(ser, reference=None):
    """Run automated calibration procedure to find optimal settings."""
    print("Starting automated calibration...")
    
    best_settings = DEFAULT_SETTINGS.copy()
    best_quality = -1
    
    # Test different resolutions
    for resolution in ["VGA", "SVGA", "XGA"]:
        set_camera_param(ser, "resolution", resolution)
        
        # Test different brightness and contrast combinations
        for brightness in [-2, 0, 2]:
            set_camera_param(ser, "brightness", brightness)
            
            for contrast in [-2, 0, 2]:
                set_camera_param(ser, "contrast", contrast)
                
                # Capture image and analyze quality
                image = capture_image(ser)
                quality_metrics = analyze_image_quality(image, reference)
                
                if quality_metrics:
                    # Simple quality score - can be refined based on specific needs
                    quality_score = quality_metrics["clarity"] * (1 - abs(quality_metrics["dark_ratio"] - 0.2)) * (1 - abs(quality_metrics["bright_ratio"] - 0.2))
                    
                    if quality_metrics["reference_similarity"]:
                        quality_score *= quality_metrics["reference_similarity"]
                    
                    print(f"Settings: resolution={resolution}, brightness={brightness}, contrast={contrast}, quality={quality_score:.2f}")
                    
                    if quality_score > best_quality:
                        best_quality = quality_score
                        best_settings["resolution"] = resolution
                        best_settings["brightness"] = brightness
                        best_settings["contrast"] = contrast
    
    # Set best settings and return
    for param, value in best_settings.items():
        set_camera_param(ser, param, value)
    
    print(f"Auto-calibration complete. Best quality score: {best_quality:.2f}")
    return best_settings

def set_camera_param(ser, param, value):
    """Set camera parameter on the ESP32."""
    command = f"SET {param} {value}"
    response = send_command(ser, command)
    
    if response and "OK" in response:
        print(f"Set {param} to {value}")
        return True
    else:
        print(f"Failed to set {param} to {value}")
        return False

def interactive_calibration(ser):
    """Interactive calibration procedure with user feedback."""
    print("\nInteractive Camera Calibration")
    print("=============================")
    print("This tool helps you fine-tune camera settings for optimal OCR performance.")
    
    current_settings = DEFAULT_SETTINGS.copy()
    
    while True:
        print("\nCurrent settings:")
        for i, (param, value) in enumerate(current_settings.items()):
            print(f"{i+1}. {param}: {value}")
        
        print("\nOptions:")
        print("c - Capture image with current settings")
        print("a - Run automated optimization")
        print("s - Save current settings")
        print("q - Quit without saving")
        
        choice = input("\nEnter option or parameter number to change: ")
        
        if choice.lower() == 'q':
            return None
        elif choice.lower() == 'c':
            image = capture_image(ser)
            if image is not None:
                cv2.imshow("Captured Image", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
                quality = analyze_image_quality(image)
                if quality:
                    print("\nImage quality metrics:")
                    for metric, value in quality.items():
                        if value is not None:
                            print(f"{metric}: {value:.4f}")
        elif choice.lower() == 'a':
            new_settings = auto_calibrate(ser)
            if new_settings:
                current_settings = new_settings
        elif choice.lower() == 's':
            return current_settings
        elif choice.isdigit() and 1 <= int(choice) <= len(current_settings):
            param = list(current_settings.keys())[int(choice) - 1]
            print(f"\nChanging {param} (current value: {current_settings[param]})")
            
            if param == "resolution":
                print("Available resolutions:")
                for i, res in enumerate(RESOLUTION_MAP.keys()):
                    print(f"{i+1}. {res} - {RESOLUTION_MAP[res]}")
                res_choice = input("Select resolution number: ")
                if res_choice.isdigit() and 1 <= int(res_choice) <= len(RESOLUTION_MAP):
                    new_value = list(RESOLUTION_MAP.keys())[int(res_choice) - 1]
                else:
                    continue
            else:
                new_value = input(f"Enter new value for {param}: ")
                try:
                    if param in ["contrast", "brightness", "saturation"]:
                        new_value = int(new_value)
                    elif param in ["awb_gain"]:
                        new_value = float(new_value)
                    elif param in ["bpc", "wpc", "raw_gma", "lens_correction", "binning"]:
                        new_value = int(new_value) if new_value in ["0", "1"] else current_settings[param]
                except ValueError:
                    print("Invalid input, keeping current value")
                    continue
            
            if set_camera_param(ser, param, new_value):
                current_settings[param] = new_value

def save_calibration(settings, output_file):
    """Save calibration settings to a file."""
    try:
        with open(output_file, 'w') as f:
            json.dump(settings, f, indent=4)
        print(f"Calibration saved to {output_file}")
        return True
    except Exception as e:
        print(f"Error saving calibration: {e}")
        return False

def main():
    """Main entry point of the script."""
    args = parse_args()
    
    # Find serial port if not specified
    port = args.port
    if not port:
        port = find_esp32_port()
        if not port:
            print("Error: Could not find ESP32 device. Please specify port with --port")
            return
    
    # Connect to device
    ser = connect_to_device(port, args.baud)
    if not ser:
        return
    
    # Load reference image if provided
    reference = None
    if args.reference:
        if os.path.exists(args.reference):
            reference = cv2.imread(args.reference)
            if reference is None:
                print(f"Error: Could not load reference image {args.reference}")
        else:
            print(f"Error: Reference image {args.reference} not found")
    
    # Run calibration
    if args.auto:
        settings = auto_calibrate(ser, reference)
    else:
        settings = interactive_calibration(ser)
    
    # Save calibration if we have settings
    if settings:
        save_calibration(settings, args.output)
    
    # Clean up
    ser.close()
    print("Calibration completed")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCalibration aborted by user")
        sys.exit(0)
