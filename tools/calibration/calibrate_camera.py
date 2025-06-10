#!/usr/bin/env python3
"""
Smart Meter Reader OCR - Camera Calibration Tool

This script helps calibrate camera settings for optimal meter reading performance.
It provides tools for:
- Camera parameter optimization (focus, exposure, white balance)
- Image quality assessment
- Real-time preview and adjustment
- Automatic calibration using test patterns
- ESP32 camera configuration generation

Features:
- Interactive camera adjustment interface
- Automatic image quality assessment
- Test pattern recognition for calibration
- ESP32 configuration file generation
- Real-time feedback and recommendations

Usage:
    # Interactive calibration
    python calibrate_camera.py --interactive

    # Automatic calibration with test pattern
    python calibrate_camera.py --auto-calibrate --test-pattern checkerboard

    # Generate ESP32 config
    python calibrate_camera.py --generate-esp32-config --output camera_config.h

Author: Frank Amoah A.K.A SiRWaTT Smart Meter Reader OCR Team Lead
"""

import argparse
import cv2
import numpy as np
import json
import time
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import serial
import serial.tools.list_ports
from dataclasses import dataclass, asdict
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import queue

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CameraSettings:
    """Data class to store camera settings"""
    brightness: int = 0
    contrast: int = 0
    saturation: int = 0
    hue: int = 0
    gain: int = 0
    exposure: int = -4
    white_balance: int = 4000
    focus: int = 0
    resolution_width: int = 1280
    resolution_height: int = 720
    frame_rate: int = 15
    quality: int = 12
    format: str = "JPEG"

@dataclass
class ImageQualityMetrics:
    """Data class to store image quality metrics"""
    sharpness: float = 0.0
    contrast: float = 0.0
    brightness: float = 0.0
    noise_level: float = 0.0
    overall_score: float = 0.0
    recommendations: List[str] = None

    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []

class CameraCalibrator:
    """Camera calibration tool for smart meter reading"""
    
    def __init__(self, output_dir: str = "./calibration_results"):
        """
        Initialize the camera calibrator
        
        Args:
            output_dir (str): Directory to save calibration results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.cap = None
        self.current_settings = CameraSettings()
        self.calibration_results = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # ESP32 serial connection
        self.esp32_serial = None
        
        logger.info(f"Camera calibrator initialized. Output directory: {self.output_dir}")

    def find_cameras(self) -> List[int]:
        """Find available cameras"""
        cameras = []
        for i in range(10):  # Check first 10 camera indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cameras.append(i)
                cap.release()
        return cameras

    def connect_camera(self, camera_id: int = 0) -> bool:
        """
        Connect to camera
        
        Args:
            camera_id (int): Camera device ID
            
        Returns:
            bool: True if successful
        """
        try:
            self.cap = cv2.VideoCapture(camera_id)
            if not self.cap.isOpened():
                return False
            
            # Set initial camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.current_settings.resolution_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.current_settings.resolution_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.current_settings.frame_rate)
            
            logger.info(f"Connected to camera {camera_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to camera {camera_id}: {e}")
            return False

    def apply_settings(self, settings: CameraSettings):
        """Apply camera settings"""
        if self.cap is None:
            return False
        
        try:
            # Apply settings that are supported by OpenCV
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, settings.brightness / 100.0)
            self.cap.set(cv2.CAP_PROP_CONTRAST, settings.contrast / 100.0)
            self.cap.set(cv2.CAP_PROP_SATURATION, settings.saturation / 100.0)
            self.cap.set(cv2.CAP_PROP_HUE, settings.hue / 180.0)
            self.cap.set(cv2.CAP_PROP_GAIN, settings.gain)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, settings.exposure)
            
            # Note: Some settings like white_balance and focus might not be available
            # on all cameras via OpenCV
            
            self.current_settings = settings
            return True
            
        except Exception as e:
            logger.warning(f"Some camera settings could not be applied: {e}")
            return False

    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame"""
        if self.cap is None:
            return None
        
        ret, frame = self.cap.read()
        return frame if ret else None

    def assess_image_quality(self, image: np.ndarray) -> ImageQualityMetrics:
        """
        Assess image quality for OCR suitability
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            ImageQualityMetrics: Quality assessment results
        """
        if image is None:
            return ImageQualityMetrics()
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # 1. Sharpness (using Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness = min(laplacian_var / 1000.0, 1.0)  # Normalize to 0-1
        
        # 2. Contrast (using standard deviation)
        contrast = min(gray.std() / 128.0, 1.0)  # Normalize to 0-1
        
        # 3. Brightness (distance from ideal range)
        mean_brightness = gray.mean()
        ideal_brightness = 128
        brightness_score = 1.0 - abs(mean_brightness - ideal_brightness) / 128.0
        
        # 4. Noise level (using median filter difference)
        filtered = cv2.medianBlur(gray, 5)
        noise_diff = np.mean(np.abs(gray.astype(float) - filtered.astype(float)))
        noise_score = max(0.0, 1.0 - noise_diff / 50.0)
        
        # Calculate overall score (weighted average)
        weights = [0.3, 0.3, 0.2, 0.2]  # sharpness, contrast, brightness, noise
        overall_score = sum(score * weight for score, weight in 
                          zip([sharpness, contrast, brightness_score, noise_score], weights))
        
        # Generate recommendations
        recommendations = []
        if sharpness < 0.3:
            recommendations.append("Image is blurry - check focus settings")
        if contrast < 0.3:
            recommendations.append("Low contrast - adjust lighting or camera contrast")
        if brightness_score < 0.5:
            recommendations.append("Poor brightness - adjust exposure or lighting")
        if noise_score < 0.5:
            recommendations.append("High noise level - reduce gain or improve lighting")
        
        return ImageQualityMetrics(
            sharpness=sharpness,
            contrast=contrast,
            brightness=brightness_score,
            noise_level=noise_score,
            overall_score=overall_score,
            recommendations=recommendations
        )

    def interactive_calibration(self):
        """Run interactive calibration with GUI"""
        logger.info("Starting interactive calibration...")
        
        # Find available cameras
        cameras = self.find_cameras()
        if not cameras:
            messagebox.showerror("Error", "No cameras found!")
            return
        
        # Connect to first available camera
        if not self.connect_camera(cameras[0]):
            messagebox.showerror("Error", "Failed to connect to camera!")
            return
        
        # Create GUI
        self.create_calibration_gui()

    def create_calibration_gui(self):
        """Create interactive calibration GUI"""
        root = tk.Tk()
        root.title("Smart Meter OCR - Camera Calibration")
        root.geometry("1200x800")
        
        # Create main frames
        left_frame = ttk.Frame(root, padding="10")
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        right_frame = ttk.Frame(root, padding="10")
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)
        
        # Camera preview (left side)
        preview_label = ttk.Label(left_frame, text="Camera Preview")
        preview_label.grid(row=0, column=0, pady=5)
        
        self.preview_canvas = tk.Canvas(left_frame, width=640, height=480, bg='black')
        self.preview_canvas.grid(row=1, column=0, pady=5)
        
        # Quality metrics display
        self.quality_text = tk.Text(left_frame, height=8, width=50)
        self.quality_text.grid(row=2, column=0, pady=5)
        
        # Control buttons
        button_frame = ttk.Frame(left_frame)
        button_frame.grid(row=3, column=0, pady=10)
        
        ttk.Button(button_frame, text="Capture Test Image", 
                  command=self.capture_test_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Auto Optimize", 
                  command=self.auto_optimize).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Settings", 
                  command=self.save_current_settings).pack(side=tk.LEFT, padx=5)
        
        # Camera controls (right side)
        controls_label = ttk.Label(right_frame, text="Camera Controls")
        controls_label.grid(row=0, column=0, columnspan=2, pady=5)
        
        # Create sliders for camera settings
        self.create_camera_controls(right_frame)
        
        # Start preview update loop
        self.update_preview()
        
        root.mainloop()

    def create_camera_controls(self, parent):
        """Create camera control sliders"""
        controls = [
            ("Brightness", "brightness", -100, 100, 0),
            ("Contrast", "contrast", -100, 100, 0),
            ("Saturation", "saturation", -100, 100, 0),
            ("Hue", "hue", -180, 180, 0),
            ("Gain", "gain", 0, 100, 0),
            ("Exposure", "exposure", -10, 0, -4),
        ]
        
        self.control_vars = {}
        row = 1
        
        for label, attr, min_val, max_val, default in controls:
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=2)
            
            var = tk.IntVar(value=default)
            self.control_vars[attr] = var
            
            scale = ttk.Scale(parent, from_=min_val, to=max_val, 
                            variable=var, orient=tk.HORIZONTAL, length=200,
                            command=lambda val, a=attr: self.on_setting_change(a, val))
            scale.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2, padx=(10, 0))
            
            # Value display
            value_label = ttk.Label(parent, text=str(default))
            value_label.grid(row=row, column=2, sticky=tk.W, padx=(10, 0))
            
            # Store reference to update label
            setattr(self, f"{attr}_label", value_label)
            
            row += 1

    def on_setting_change(self, setting_name: str, value: str):
        """Handle setting changes from GUI"""
        try:
            int_value = int(float(value))
            
            # Update the current settings
            setattr(self.current_settings, setting_name, int_value)
            
            # Update the display label
            label = getattr(self, f"{setting_name}_label", None)
            if label:
                label.config(text=str(int_value))
            
            # Apply settings to camera
            self.apply_settings(self.current_settings)
            
        except (ValueError, AttributeError) as e:
            logger.warning(f"Error updating setting {setting_name}: {e}")

    def update_preview(self):
        """Update camera preview and quality metrics"""
        if self.cap is None:
            return
        
        try:
            frame = self.capture_frame()
            if frame is not None:
                # Resize for display
                display_frame = cv2.resize(frame, (640, 480))
                
                # Convert to RGB for tkinter
                rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image and then to PhotoImage
                from PIL import Image, ImageTk
                pil_image = Image.fromarray(rgb_frame)
                photo = ImageTk.PhotoImage(pil_image)
                
                # Update canvas
                self.preview_canvas.delete("all")
                self.preview_canvas.create_image(320, 240, image=photo)
                self.preview_canvas.image = photo  # Keep a reference
                
                # Update quality metrics
                metrics = self.assess_image_quality(frame)
                self.update_quality_display(metrics)
        
        except Exception as e:
            logger.warning(f"Error updating preview: {e}")
        
        # Schedule next update
        if hasattr(self, 'preview_canvas'):
            self.preview_canvas.after(100, self.update_preview)

    def update_quality_display(self, metrics: ImageQualityMetrics):
        """Update quality metrics display"""
        if not hasattr(self, 'quality_text'):
            return
        
        self.quality_text.delete(1.0, tk.END)
        
        quality_info = f"""Image Quality Assessment:

Overall Score: {metrics.overall_score:.3f}
Sharpness: {metrics.sharpness:.3f}
Contrast: {metrics.contrast:.3f}
Brightness: {metrics.brightness:.3f}
Noise Level: {metrics.noise_level:.3f}

Recommendations:
"""
        
        for rec in metrics.recommendations:
            quality_info += f"• {rec}\n"
        
        if not metrics.recommendations:
            quality_info += "• Image quality is good for OCR"
        
        self.quality_text.insert(1.0, quality_info)

    def capture_test_image(self):
        """Capture and save a test image"""
        if self.cap is None:
            messagebox.showerror("Error", "No camera connected!")
            return
        
        frame = self.capture_frame()
        if frame is None:
            messagebox.showerror("Error", "Failed to capture image!")
            return
        
        # Save image with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_image_{timestamp}.jpg"
        filepath = self.output_dir / filename
        
        cv2.imwrite(str(filepath), frame)
        
        # Assess quality and save report
        metrics = self.assess_image_quality(frame)
        report = {
            "timestamp": timestamp,
            "filename": filename,
            "camera_settings": asdict(self.current_settings),
            "quality_metrics": asdict(metrics)
        }
        
        report_file = self.output_dir / f"test_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        messagebox.showinfo("Success", f"Test image saved: {filename}\nQuality score: {metrics.overall_score:.3f}")

    def auto_optimize(self):
        """Automatically optimize camera settings"""
        if self.cap is None:
            messagebox.showerror("Error", "No camera connected!")
            return
        
        messagebox.showinfo("Auto Optimization", "Starting automatic optimization...\nThis may take a few minutes.")
        
        # Run optimization in separate thread to avoid blocking GUI
        threading.Thread(target=self._run_auto_optimization, daemon=True).start()

    def _run_auto_optimization(self):
        """Run automatic optimization process"""
        logger.info("Starting automatic optimization...")
        
        best_settings = self.current_settings
        best_score = 0.0
        
        # Parameters to optimize and their ranges
        param_ranges = {
            'brightness': range(-50, 51, 25),
            'contrast': range(-50, 51, 25),
            'exposure': range(-8, 1, 2),
            'gain': range(0, 51, 25)
        }
        
        total_combinations = 1
        for param_range in param_ranges.values():
            total_combinations *= len(param_range)
        
        combination_count = 0
        
        # Try different combinations
        for brightness in param_ranges['brightness']:
            for contrast in param_ranges['contrast']:
                for exposure in param_ranges['exposure']:
                    for gain in param_ranges['gain']:
                        combination_count += 1
                        
                        # Create test settings
                        test_settings = CameraSettings(
                            brightness=brightness,
                            contrast=contrast,
                            exposure=exposure,
                            gain=gain,
                            saturation=self.current_settings.saturation,
                            hue=self.current_settings.hue,
                            white_balance=self.current_settings.white_balance,
                            focus=self.current_settings.focus,
                            resolution_width=self.current_settings.resolution_width,
                            resolution_height=self.current_settings.resolution_height,
                            frame_rate=self.current_settings.frame_rate,
                            quality=self.current_settings.quality,
                            format=self.current_settings.format
                        )
                        
                        # Apply settings and capture image
                        self.apply_settings(test_settings)
                        time.sleep(0.5)  # Allow camera to adjust
                        
                        frame = self.capture_frame()
                        if frame is None:
                            continue
                        
                        # Assess quality
                        metrics = self.assess_image_quality(frame)
                        
                        # Update best settings if this is better
                        if metrics.overall_score > best_score:
                            best_score = metrics.overall_score
                            best_settings = test_settings
                        
                        logger.info(f"Optimization progress: {combination_count}/{total_combinations} "
                                  f"(Best score: {best_score:.3f})")
        
        # Apply best settings
        logger.info(f"Optimization completed. Best score: {best_score:.3f}")
        self.apply_settings(best_settings)
        self.current_settings = best_settings
        
        # Update GUI controls
        self._update_gui_controls()
        
        # Show completion message
        messagebox.showinfo("Optimization Complete", 
                          f"Optimization completed!\nBest quality score: {best_score:.3f}")

    def _update_gui_controls(self):
        """Update GUI controls to match current settings"""
        if hasattr(self, 'control_vars'):
            for attr, var in self.control_vars.items():
                value = getattr(self.current_settings, attr)
                var.set(value)
                
                # Update label
                label = getattr(self, f"{attr}_label", None)
                if label:
                    label.config(text=str(value))

    def save_current_settings(self):
        """Save current camera settings"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Capture current frame for quality assessment
        frame = self.capture_frame()
        metrics = self.assess_image_quality(frame) if frame is not None else ImageQualityMetrics()
        
        settings_data = {
            "timestamp": timestamp,
            "session_id": self.session_id,
            "camera_settings": asdict(self.current_settings),
            "quality_metrics": asdict(metrics)
        }
        
        settings_file = self.output_dir / f"camera_settings_{timestamp}.json"
        with open(settings_file, 'w') as f:
            json.dump(settings_data, f, indent=2)
        
        messagebox.showinfo("Settings Saved", f"Camera settings saved to: {settings_file}")
        logger.info(f"Camera settings saved: {settings_file}")

    def generate_esp32_config(self, output_file: str = None):
        """Generate ESP32 camera configuration header file"""
        if output_file is None:
            output_file = self.output_dir / "camera_config.h"
        
        config_content = f"""/*
 * Smart Meter Reader OCR - Camera Configuration
 * Generated: {datetime.now().isoformat()}
 * Session ID: {self.session_id}
 */

#ifndef CAMERA_CONFIG_H
#define CAMERA_CONFIG_H

#include "esp_camera.h"

// Camera pin definitions (adjust for your hardware)
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// Camera configuration
static camera_config_t camera_config = {{
    .pin_pwdn  = PWDN_GPIO_NUM,
    .pin_reset = RESET_GPIO_NUM,
    .pin_xclk = XCLK_GPIO_NUM,
    .pin_sscb_sda = SIOD_GPIO_NUM,
    .pin_sscb_scl = SIOC_GPIO_NUM,
    .pin_d7 = Y9_GPIO_NUM,
    .pin_d6 = Y8_GPIO_NUM,
    .pin_d5 = Y7_GPIO_NUM,
    .pin_d4 = Y6_GPIO_NUM,
    .pin_d3 = Y5_GPIO_NUM,
    .pin_d2 = Y4_GPIO_NUM,
    .pin_d1 = Y3_GPIO_NUM,
    .pin_d0 = Y2_GPIO_NUM,
    .pin_vsync = VSYNC_GPIO_NUM,
    .pin_href = HREF_GPIO_NUM,
    .pin_pclk = PCLK_GPIO_NUM,
    
    .xclk_freq_hz = 20000000,
    .ledc_timer = LEDC_TIMER_0,
    .ledc_channel = LEDC_CHANNEL_0,
    
    .pixel_format = PIXFORMAT_JPEG,
    .frame_size = FRAMESIZE_SXGA,    // {self.current_settings.resolution_width}x{self.current_settings.resolution_height}
    
    .jpeg_quality = {self.current_settings.quality},
    .fb_count = 1,
    .grab_mode = CAMERA_GRAB_WHEN_EMPTY
}};

// Optimized sensor settings (apply after camera init)
static void apply_optimized_settings(sensor_t *s) {{
    // Brightness: {self.current_settings.brightness}
    s->set_brightness(s, {self.current_settings.brightness});
    
    // Contrast: {self.current_settings.contrast}  
    s->set_contrast(s, {self.current_settings.contrast});
    
    // Saturation: {self.current_settings.saturation}
    s->set_saturation(s, {self.current_settings.saturation});
    
    // White balance mode
    s->set_whitebal(s, 1);  // Enable auto white balance
    
    // Gain control
    s->set_gain_ctrl(s, 1);  // Enable AGC
    s->set_agc_gain(s, {self.current_settings.gain});
    
    // Exposure control  
    s->set_exposure_ctrl(s, 1);  // Enable AEC
    s->set_aec_value(s, {abs(self.current_settings.exposure) * 100});
    
    // Special effects
    s->set_special_effect(s, 0);  // No special effects
    
    // AWB gain
    s->set_awb_gain(s, 1);  // Enable AWB gain
    
    // BPC (Black Pixel Correction)
    s->set_bpc(s, 1);  // Enable
    
    // WPC (White Pixel Correction)  
    s->set_wpc(s, 1);  // Enable
    
    // Lens correction
    s->set_lenc(s, 1);  // Enable lens correction
}}

#endif // CAMERA_CONFIG_H
"""
        
        with open(output_file, 'w') as f:
            f.write(config_content)
        
        logger.info(f"ESP32 camera configuration generated: {output_file}")
        return str(output_file)

    def cleanup(self):
        """Clean up resources"""
        if self.cap is not None:
            self.cap.release()
        if self.esp32_serial is not None:
            self.esp32_serial.close()
        cv2.destroyAllWindows()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Smart Meter OCR Camera Calibration Tool")
    
    # Mode selection
    parser.add_argument("--interactive", action="store_true", 
                       help="Run interactive calibration GUI")
    parser.add_argument("--auto-calibrate", action="store_true",
                       help="Run automatic calibration")
    
    # Camera selection
    parser.add_argument("--camera-id", type=int, default=0,
                       help="Camera device ID")
    
    # Output options
    parser.add_argument("--output-dir", default="./calibration_results",
                       help="Output directory for results")
    parser.add_argument("--generate-esp32-config", action="store_true",
                       help="Generate ESP32 camera configuration")
    parser.add_argument("--config-output", 
                       help="Output file for ESP32 configuration")
    
    # Test patterns
    parser.add_argument("--test-pattern", choices=["checkerboard", "grid"],
                       help="Test pattern for automatic calibration")
    
    return parser.parse_args()


def main():
    """Main calibration function"""
    args = parse_args()
    
    try:
        # Initialize calibrator
        calibrator = CameraCalibrator(args.output_dir)
        
        if args.interactive:
            # Run interactive calibration
            calibrator.interactive_calibration()
        elif args.auto_calibrate:
            # Run automatic calibration
            logger.info("Automatic calibration not yet implemented")
        elif args.generate_esp32_config:
            # Generate ESP32 configuration only
            output_file = args.config_output or "camera_config.h"
            calibrator.generate_esp32_config(output_file)
            print(f"ESP32 configuration generated: {output_file}")
        else:
            print("Please specify --interactive or --auto-calibrate or --generate-esp32-config")
            return 1
        
        # Cleanup
        calibrator.cleanup()
        
        logger.info("Camera calibration completed successfully!")
        
    except Exception as e:
        logger.error(f"Camera calibration failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
