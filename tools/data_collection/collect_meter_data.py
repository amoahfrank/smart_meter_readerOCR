#!/usr/bin/env python3
"""
Smart Meter Reader OCR - Data Collection Tool

This script helps collect and organize training data for the digit recognition model.
It can collect data from various sources: webcam, uploaded images, or synthetic generation.

Features:
- Real-time data collection from webcam
- Batch processing of uploaded images
- Data labeling interface
- Data augmentation
- Quality validation
- Export to training format

Usage:
    python collect_meter_data.py --source webcam --count 100 --output ./training_data
    python collect_meter_data.py --source import --input-dir ./my_images --output ./training_data
    python collect_meter_data.py --source synthetic --count 1000 --output ./training_data

Author: Smart Meter Reader OCR Team
"""

import argparse
import cv2
import numpy as np
import os
import json
import time
from datetime import datetime
from pathlib import Path
import logging
from typing import List, Tuple, Dict, Optional
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import threading
import queue

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MeterDataCollector:
    """Main class for collecting meter reading training data"""
    
    def __init__(self, output_dir: str):
        """
        Initialize the data collector
        
        Args:
            output_dir (str): Directory to save collected data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels" 
        self.preprocessed_dir = self.output_dir / "preprocessed"
        self.metadata_dir = self.output_dir / "metadata"
        
        for dir_path in [self.images_dir, self.labels_dir, self.preprocessed_dir, self.metadata_dir]:
            dir_path.mkdir(exist_ok=True)
            
        # Data tracking
        self.collected_count = 0
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load existing metadata if available
        self.metadata_file = self.metadata_dir / "collection_metadata.json"
        self.metadata = self.load_metadata()
        
        logger.info(f"Data collector initialized. Output directory: {self.output_dir}")
        logger.info(f"Session ID: {self.session_id}")

    def load_metadata(self) -> Dict:
        """Load existing metadata or create new"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        else:
            return {
                "sessions": {},
                "total_images": 0,
                "creation_date": datetime.now().isoformat(),
                "data_sources": {}
            }

    def save_metadata(self):
        """Save metadata to file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def collect_from_webcam(self, target_count: int, camera_id: int = 0):
        """
        Collect data from webcam with interactive labeling
        
        Args:
            target_count (int): Number of images to collect
            camera_id (int): Camera device ID
        """
        logger.info(f"Starting webcam collection. Target: {target_count} images")
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Cannot open camera {camera_id}")
            
        # Set camera properties for better quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        
        collected = 0
        session_data = {
            "source": "webcam",
            "camera_id": camera_id,
            "start_time": datetime.now().isoformat(),
            "images": []
        }
        
        print("\\n=== Webcam Data Collection ===")
        print("Instructions:")
        print("- Press SPACE to capture image")
        print("- Press 'q' to quit")
        print("- Position meter display clearly in the frame")
        print("- Ensure good lighting and focus")
        print(f"Target: {target_count} images\\n")
        
        try:
            while collected < target_count:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read from camera")
                    break
                
                # Display frame with overlay information
                display_frame = frame.copy()
                self.add_capture_overlay(display_frame, collected, target_count)
                
                cv2.imshow('Data Collection - Press SPACE to capture, Q to quit', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):  # Space to capture
                    # Save image
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    image_filename = f"webcam_{self.session_id}_{timestamp}.jpg"
                    image_path = self.images_dir / image_filename
                    
                    cv2.imwrite(str(image_path), frame)
                    
                    # Get user label
                    label = self.get_user_label_gui(frame)
                    if label is not None:
                        # Save label
                        label_data = {
                            "filename": image_filename,
                            "reading": label,
                            "timestamp": timestamp,
                            "source": "webcam",
                            "session_id": self.session_id,
                            "quality_score": self.assess_image_quality(frame)
                        }
                        
                        label_path = self.labels_dir / f"{image_filename.replace('.jpg', '.json')}"
                        with open(label_path, 'w') as f:
                            json.dump(label_data, f, indent=2)
                        
                        session_data["images"].append(label_data)
                        collected += 1
                        
                        logger.info(f"Captured image {collected}/{target_count}: {label}")
                    else:
                        # User cancelled, remove image
                        os.remove(image_path)
                        
                elif key == ord('q'):  # Quit
                    break
                    
        except KeyboardInterrupt:
            logger.info("Collection interrupted by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
        session_data["end_time"] = datetime.now().isoformat()
        session_data["collected_count"] = collected
        self.metadata["sessions"][self.session_id] = session_data
        self.save_metadata()
        
        logger.info(f"Webcam collection completed. Collected {collected} images")

    def collect_from_directory(self, input_dir: str):
        """
        Process existing images from directory
        
        Args:
            input_dir (str): Directory containing images to process
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")
            
        logger.info(f"Processing images from directory: {input_dir}")
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(input_path.glob(f"*{ext}")))
            image_files.extend(list(input_path.glob(f"*{ext.upper()}")))
            
        if not image_files:
            logger.warning("No image files found in input directory")
            return
            
        logger.info(f"Found {len(image_files)} image files")
        
        session_data = {
            "source": "directory",
            "input_directory": str(input_path),
            "start_time": datetime.now().isoformat(),
            "images": []
        }
        
        processed = 0
        for image_file in image_files:
            try:
                # Load and validate image
                image = cv2.imread(str(image_file))
                if image is None:
                    logger.warning(f"Could not load image: {image_file}")
                    continue
                
                # Copy to our dataset
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                new_filename = f"imported_{self.session_id}_{timestamp}_{image_file.name}"
                new_image_path = self.images_dir / new_filename
                
                cv2.imwrite(str(new_image_path), image)
                
                # Get label from user
                label = self.get_user_label_gui(image, f"Label for {image_file.name}")
                if label is not None:
                    label_data = {
                        "filename": new_filename,
                        "reading": label,
                        "timestamp": timestamp,
                        "source": "directory_import",
                        "original_file": str(image_file),
                        "session_id": self.session_id,
                        "quality_score": self.assess_image_quality(image)
                    }
                    
                    label_path = self.labels_dir / f"{new_filename.replace('.jpg', '.json')}"
                    with open(label_path, 'w') as f:
                        json.dump(label_data, f, indent=2)
                    
                    session_data["images"].append(label_data)
                    processed += 1
                    
                    logger.info(f"Processed {processed}: {image_file.name} -> {label}")
                else:
                    # User cancelled, remove copied image
                    os.remove(new_image_path)
                    
            except Exception as e:
                logger.error(f"Error processing {image_file}: {e}")
                continue
                
        session_data["end_time"] = datetime.now().isoformat() 
        session_data["processed_count"] = processed
        self.metadata["sessions"][self.session_id] = session_data
        self.save_metadata()
        
        logger.info(f"Directory processing completed. Processed {processed} images")

    def generate_synthetic_data(self, count: int):
        """
        Generate synthetic meter reading images
        
        Args:
            count (int): Number of synthetic images to generate
        """
        logger.info(f"Generating {count} synthetic images")
        
        session_data = {
            "source": "synthetic",
            "start_time": datetime.now().isoformat(),
            "images": []
        }
        
        # Load fonts for rendering digits
        font_paths = self.find_digit_fonts()
        
        for i in range(count):
            try:
                # Generate random reading
                reading = self.generate_random_reading()
                
                # Create synthetic image
                image = self.create_synthetic_meter_display(reading, font_paths)
                
                # Add noise and variations
                image = self.add_synthetic_variations(image)
                
                # Save image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filename = f"synthetic_{self.session_id}_{timestamp}_{i:04d}.jpg"
                image_path = self.images_dir / filename
                
                cv2.imwrite(str(image_path), image)
                
                # Save label
                label_data = {
                    "filename": filename,
                    "reading": reading,
                    "timestamp": timestamp,
                    "source": "synthetic",
                    "session_id": self.session_id,
                    "quality_score": 1.0  # Synthetic images have known quality
                }
                
                label_path = self.labels_dir / f"{filename.replace('.jpg', '.json')}"
                with open(label_path, 'w') as f:
                    json.dump(label_data, f, indent=2)
                
                session_data["images"].append(label_data)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Generated {i + 1}/{count} synthetic images")
                    
            except Exception as e:
                logger.error(f"Error generating synthetic image {i}: {e}")
                continue
                
        session_data["end_time"] = datetime.now().isoformat()
        session_data["generated_count"] = count
        self.metadata["sessions"][self.session_id] = session_data
        self.save_metadata()
        
        logger.info(f"Synthetic generation completed. Generated {count} images")

    def add_capture_overlay(self, frame: np.ndarray, current: int, target: int):
        """Add overlay information to capture frame"""
        height, width = frame.shape[:2]
        
        # Add semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"Collected: {current}/{target}", (20, 35), font, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "SPACE: Capture | Q: Quit", (20, 60), font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Position meter display in center", (20, 80), font, 0.5, (255, 255, 0), 1)
        
        # Add crosshairs for alignment
        center_x, center_y = width // 2, height // 2
        cv2.line(frame, (center_x - 50, center_y), (center_x + 50, center_y), (0, 255, 0), 1)
        cv2.line(frame, (center_x, center_y - 50), (center_x, center_y + 50), (0, 255, 0), 1)

    def get_user_label_gui(self, image: np.ndarray, title: str = "Enter Reading") -> Optional[str]:
        """
        Show GUI for user to enter the meter reading label
        
        Args:
            image (np.ndarray): Image to display for labeling
            title (str): Window title
            
        Returns:
            Optional[str]: User-entered label or None if cancelled
        """
        result = queue.Queue()
        
        def create_gui():
            root = tk.Tk()
            root.title(title)
            root.geometry("800x600")
            
            # Convert OpenCV image to PIL for display
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # Resize image to fit window
            display_size = (400, 300)
            pil_image.thumbnail(display_size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(pil_image)
            
            # Create GUI elements
            main_frame = ttk.Frame(root, padding="10")
            main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            
            # Image display
            image_label = ttk.Label(main_frame, image=photo)
            image_label.grid(row=0, column=0, columnspan=2, pady=10)
            
            # Instructions
            instruction_label = ttk.Label(main_frame, 
                text="Enter the meter reading shown in the image (digits only):",
                font=('Arial', 12))
            instruction_label.grid(row=1, column=0, columnspan=2, pady=5)
            
            # Entry field
            entry_var = tk.StringVar()
            entry_field = ttk.Entry(main_frame, textvariable=entry_var, font=('Arial', 14), width=20)
            entry_field.grid(row=2, column=0, columnspan=2, pady=10)
            entry_field.focus()
            
            # Buttons
            button_frame = ttk.Frame(main_frame)
            button_frame.grid(row=3, column=0, columnspan=2, pady=10)
            
            def on_submit():
                reading = entry_var.get().strip()
                if reading and reading.isdigit():
                    result.put(reading)
                    root.destroy()
                else:
                    messagebox.showerror("Invalid Input", "Please enter digits only (0-9)")
            
            def on_cancel():
                result.put(None)
                root.destroy()
            
            def on_enter(event):
                on_submit()
            
            ttk.Button(button_frame, text="Submit", command=on_submit).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side=tk.LEFT, padx=5)
            
            # Bind Enter key
            root.bind('<Return>', on_enter)
            entry_field.bind('<Return>', on_enter)
            
            # Keep reference to photo to prevent garbage collection
            root.photo = photo
            
            root.mainloop()
        
        # Run GUI in separate thread to avoid blocking
        gui_thread = threading.Thread(target=create_gui)
        gui_thread.daemon = True
        gui_thread.start()
        gui_thread.join()
        
        try:
            return result.get_nowait()
        except queue.Empty:
            return None

    def assess_image_quality(self, image: np.ndarray) -> float:
        """
        Assess the quality of an image for OCR
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            float: Quality score between 0 and 1
        """
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Calculate various quality metrics
        scores = []
        
        # 1. Sharpness (using Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(laplacian_var / 1000.0, 1.0)  # Normalize
        scores.append(sharpness_score)
        
        # 2. Contrast (using standard deviation)
        contrast_score = min(gray.std() / 128.0, 1.0)  # Normalize
        scores.append(contrast_score)
        
        # 3. Brightness (distance from ideal range)
        mean_brightness = gray.mean()
        ideal_brightness = 128
        brightness_score = 1.0 - abs(mean_brightness - ideal_brightness) / 128.0
        scores.append(brightness_score)
        
        # 4. Noise level (using median filter difference)
        filtered = cv2.medianBlur(gray, 5)
        noise_diff = np.mean(np.abs(gray.astype(float) - filtered.astype(float)))
        noise_score = max(0.0, 1.0 - noise_diff / 50.0)
        scores.append(noise_score)
        
        # Overall quality is weighted average
        weights = [0.3, 0.3, 0.2, 0.2]  # Prioritize sharpness and contrast
        quality_score = sum(score * weight for score, weight in zip(scores, weights))
        
        return max(0.0, min(1.0, quality_score))

    def find_digit_fonts(self) -> List[str]:
        """Find available fonts suitable for digit rendering"""
        # Common font paths on different systems
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
            "/System/Library/Fonts/Arial.ttf",  # macOS
            "C:/Windows/Fonts/arial.ttf",  # Windows
            "/usr/share/fonts/TTF/arial.ttf",  # Some Linux distributions
        ]
        
        available_fonts = []
        for font_path in font_paths:
            if os.path.exists(font_path):
                available_fonts.append(font_path)
                
        # If no system fonts found, use default
        if not available_fonts:
            available_fonts = [None]  # PIL will use default font
            
        return available_fonts

    def generate_random_reading(self) -> str:
        """Generate a random meter reading"""
        # Realistic meter reading patterns
        patterns = [
            lambda: f"{np.random.randint(10000, 99999)}",  # 5 digits
            lambda: f"{np.random.randint(100000, 999999)}",  # 6 digits  
            lambda: f"{np.random.randint(1000, 9999)}",  # 4 digits
            lambda: f"{np.random.randint(10, 999):03d}",  # 3 digits with leading zeros
        ]
        
        pattern = np.random.choice(patterns)
        return pattern()

    def create_synthetic_meter_display(self, reading: str, font_paths: List[str]) -> np.ndarray:
        """
        Create synthetic meter display image
        
        Args:
            reading (str): The reading to display
            font_paths (List[str]): Available font paths
            
        Returns:
            np.ndarray: Synthetic meter display image
        """
        # Image dimensions
        width, height = 400, 200
        
        # Create PIL image for text rendering
        pil_image = Image.new('RGB', (width, height), color=(240, 240, 240))
        draw = ImageDraw.Draw(pil_image)
        
        # Choose random font
        font_path = np.random.choice(font_paths)
        try:
            if font_path is not None:
                font_size = np.random.randint(40, 80)
                font = ImageFont.truetype(font_path, font_size)
            else:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        # Get text dimensions
        bbox = draw.textbbox((0, 0), reading, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Center text
        x = (width - text_width) // 2
        y = (height - text_height) // 2
        
        # Choose random text color (dark colors for contrast)
        text_color = (
            np.random.randint(0, 80),  # Dark colors
            np.random.randint(0, 80),
            np.random.randint(0, 80)
        )
        
        # Draw text
        draw.text((x, y), reading, fill=text_color, font=font)
        
        # Add meter display frame/border
        frame_color = (100, 100, 100)
        frame_width = 3
        draw.rectangle([frame_width, frame_width, width-frame_width, height-frame_width], 
                      outline=frame_color, width=frame_width)
        
        # Convert back to OpenCV format
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return cv_image

    def add_synthetic_variations(self, image: np.ndarray) -> np.ndarray:
        """
        Add realistic variations to synthetic images
        
        Args:
            image (np.ndarray): Input synthetic image
            
        Returns:
            np.ndarray: Image with added variations
        """
        # Random geometric transformations
        if np.random.random() < 0.3:  # 30% chance
            # Slight rotation
            angle = np.random.uniform(-5, 5)
            center = (image.shape[1] // 2, image.shape[0] // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), 
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(240, 240, 240))
        
        # Add noise
        if np.random.random() < 0.4:  # 40% chance
            noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
            image = cv2.add(image, noise)
        
        # Brightness variation
        if np.random.random() < 0.5:  # 50% chance
            brightness = np.random.uniform(0.7, 1.3)
            image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
        
        # Blur (simulate camera focus issues)
        if np.random.random() < 0.2:  # 20% chance
            kernel_size = np.random.choice([3, 5])
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        return image

    def export_training_data(self, train_split: float = 0.8, val_split: float = 0.1):
        """
        Export collected data in training format
        
        Args:
            train_split (float): Fraction of data for training
            val_split (float): Fraction of data for validation (rest goes to test)
        """
        logger.info("Exporting data for training...")
        
        # Get all labeled images
        label_files = list(self.labels_dir.glob("*.json"))
        if not label_files:
            logger.warning("No labeled data found")
            return
            
        # Load all labels
        all_data = []
        for label_file in label_files:
            with open(label_file, 'r') as f:
                label_data = json.load(f)
                image_path = self.images_dir / label_data["filename"]
                if image_path.exists():
                    all_data.append({
                        "image_path": str(image_path),
                        "label": label_data["reading"],
                        "metadata": label_data
                    })
        
        if not all_data:
            logger.warning("No valid image-label pairs found")
            return
            
        # Shuffle data
        np.random.shuffle(all_data)
        
        # Split data
        n_total = len(all_data)
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)
        n_test = n_total - n_train - n_val
        
        train_data = all_data[:n_train]
        val_data = all_data[n_train:n_train + n_val]
        test_data = all_data[n_train + n_val:]
        
        # Create export directory
        export_dir = self.output_dir / "training_export"
        export_dir.mkdir(exist_ok=True)
        
        # Export splits
        splits = {
            "train": train_data,
            "validation": val_data,
            "test": test_data
        }
        
        for split_name, split_data in splits.items():
            split_file = export_dir / f"{split_name}.json"
            with open(split_file, 'w') as f:
                json.dump(split_data, f, indent=2)
            logger.info(f"Exported {len(split_data)} samples to {split_name} set")
        
        # Export summary
        summary = {
            "total_samples": n_total,
            "train_samples": n_train,
            "validation_samples": n_val,
            "test_samples": n_test,
            "export_date": datetime.now().isoformat(),
            "data_sources": list(set(item["metadata"]["source"] for item in all_data))
        }
        
        summary_file = export_dir / "export_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Training data export completed: {export_dir}")
        logger.info(f"Total: {n_total}, Train: {n_train}, Val: {n_val}, Test: {n_test}")


def main():
    """Main function to handle command line arguments and run data collection"""
    parser = argparse.ArgumentParser(description="Smart Meter Reader OCR - Data Collection Tool")
    
    parser.add_argument("--source", choices=["webcam", "import", "synthetic"], required=True,
                       help="Data source: webcam for live capture, import for existing images, synthetic for generated data")
    parser.add_argument("--output", required=True,
                       help="Output directory for collected data")
    parser.add_argument("--count", type=int, default=100,
                       help="Number of images to collect (for webcam and synthetic)")
    parser.add_argument("--input-dir", 
                       help="Input directory for importing existing images")
    parser.add_argument("--camera-id", type=int, default=0,
                       help="Camera device ID for webcam capture")
    parser.add_argument("--export", action="store_true",
                       help="Export collected data for training after collection")
    parser.add_argument("--train-split", type=float, default=0.8,
                       help="Training data split ratio (default: 0.8)")
    parser.add_argument("--val-split", type=float, default=0.1,
                       help="Validation data split ratio (default: 0.1)")
    
    args = parser.parse_args()
    
    try:
        # Initialize collector
        collector = MeterDataCollector(args.output)
        
        # Run collection based on source
        if args.source == "webcam":
            collector.collect_from_webcam(args.count, args.camera_id)
        elif args.source == "import":
            if not args.input_dir:
                parser.error("--input-dir is required when using import source")
            collector.collect_from_directory(args.input_dir)
        elif args.source == "synthetic":
            collector.generate_synthetic_data(args.count)
            
        # Export for training if requested
        if args.export:
            collector.export_training_data(args.train_split, args.val_split)
            
        logger.info("Data collection completed successfully!")
        
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main())
