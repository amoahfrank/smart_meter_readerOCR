#!/usr/bin/env python3
"""
Smart Meter Reader OCR - Meter Display Simulator

This script simulates various types of meter displays for testing and development.
It can generate realistic meter readings with different display types, fonts,
backgrounds, and environmental conditions.

Features:
- Multiple meter display types (LCD, LED, mechanical)
- Realistic fonts and styling
- Environmental variations (lighting, shadows, reflections)
- Batch generation for training data
- Real-time simulation for testing
- Export in various formats

Usage:
    # Generate LCD digital meter displays
    python meter_simulator.py --type lcd_digital --count 100 --output ./simulated_meters

    # Generate mechanical counter displays
    python meter_simulator.py --type mechanical --count 50 --output ./simulated_meters

    # Real-time simulation window
    python meter_simulator.py --realtime --type led_display

Author: Smart Meter Reader OCR Team
"""

import argparse
import cv2
import numpy as np
import random
import json
import time
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MeterDisplaySimulator:
    """Simulator for various types of meter displays"""
    
    def __init__(self, output_dir: str = "./simulated_meters"):
        """
        Initialize the meter display simulator
        
        Args:
            output_dir (str): Directory to save simulated images
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        self.metadata_dir = self.output_dir / "metadata"
        
        for dir_path in [self.images_dir, self.labels_dir, self.metadata_dir]:
            dir_path.mkdir(exist_ok=True)
        
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load fonts
        self.fonts = self.load_fonts()
        
        # Meter display configurations
        self.display_configs = {
            'lcd_digital': {
                'background_color': (200, 220, 200),
                'text_color': (20, 20, 20),
                'frame_color': (150, 150, 150),
                'segment_style': 'lcd'
            },
            'led_display': {
                'background_color': (20, 20, 20),
                'text_color': (255, 50, 50),
                'frame_color': (80, 80, 80),
                'segment_style': 'led'
            },
            'mechanical': {
                'background_color': (240, 240, 240),
                'text_color': (30, 30, 30),
                'frame_color': (100, 100, 100),
                'segment_style': 'mechanical'
            },
            'analog_dial': {
                'background_color': (250, 250, 250),
                'text_color': (0, 0, 0),
                'frame_color': (120, 120, 120),
                'segment_style': 'analog'
            }
        }
        
        logger.info(f"Meter simulator initialized. Output directory: {self.output_dir}")

    def load_fonts(self) -> Dict[str, str]:
        """Load available fonts for meter displays"""
        font_paths = {
            'digital': [
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
                "/System/Library/Fonts/Courier.ttc",
                "C:/Windows/Fonts/consola.ttf",
                None  # Default font
            ],
            'mechanical': [
                "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                "/System/Library/Fonts/Arial.ttf",
                "C:/Windows/Fonts/arial.ttf",
                None  # Default font
            ]
        }
        
        available_fonts = {}
        for font_type, paths in font_paths.items():
            for font_path in paths:
                if font_path is None or Path(font_path).exists():
                    available_fonts[font_type] = font_path
                    break
            
            if font_type not in available_fonts:
                available_fonts[font_type] = None  # Use default
        
        return available_fonts

    def generate_reading(self, digit_count: int = 5, reading_type: str = "sequential") -> str:
        """
        Generate a realistic meter reading
        
        Args:
            digit_count (int): Number of digits in the reading
            reading_type (str): Type of reading pattern
            
        Returns:
            str: Generated reading
        """
        if reading_type == "sequential":
            # Generate realistic sequential reading
            base = random.randint(10**(digit_count-2), 10**(digit_count-1))
            return f"{base:0{digit_count}d}"
        elif reading_type == "random":
            # Completely random reading
            return ''.join([str(random.randint(0, 9)) for _ in range(digit_count)])
        elif reading_type == "realistic":
            # More realistic utility meter reading
            # Most utility readings don't start with 0 and have realistic progression
            first_digits = random.randint(1, 9)
            remaining = digit_count - 1
            if remaining > 0:
                rest = random.randint(10**(remaining-1) if remaining > 1 else 0, 
                                    10**remaining - 1)
                return f"{first_digits}{rest:0{remaining}d}"
            else:
                return str(first_digits)
        
        return "12345"  # Default fallback

    def create_lcd_display(self, reading: str, width: int = 400, height: int = 150) -> Image.Image:
        """Create LCD-style digital display"""
        config = self.display_configs['lcd_digital']
        
        # Create base image
        image = Image.new('RGB', (width, height), config['background_color'])
        draw = ImageDraw.Draw(image)
        
        # Draw frame
        frame_width = 8
        draw.rectangle([0, 0, width-1, height-1], 
                      outline=config['frame_color'], width=frame_width)
        
        # Draw inner frame (LCD screen area)
        inner_margin = 20
        draw.rectangle([inner_margin, inner_margin, width-inner_margin, height-inner_margin],
                      fill=(180, 200, 180), outline=(120, 140, 120), width=2)
        
        # Calculate font size and position
        font_size = min(height // 3, width // (len(reading) + 2))
        try:
            font = ImageFont.truetype(self.fonts['digital'], font_size) if self.fonts['digital'] else ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        # Get text dimensions
        bbox = draw.textbbox((0, 0), reading, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Center text
        x = (width - text_width) // 2
        y = (height - text_height) // 2
        
        # Draw reading with LCD effect
        # Shadow effect
        draw.text((x + 2, y + 2), reading, fill=(100, 120, 100), font=font)
        # Main text
        draw.text((x, y), reading, fill=config['text_color'], font=font)
        
        return image

    def create_led_display(self, reading: str, width: int = 400, height: int = 150) -> Image.Image:
        """Create LED-style display"""
        config = self.display_configs['led_display']
        
        # Create base image
        image = Image.new('RGB', (width, height), config['background_color'])
        draw = ImageDraw.Draw(image)
        
        # Draw frame
        frame_width = 6
        draw.rectangle([0, 0, width-1, height-1], 
                      outline=config['frame_color'], width=frame_width)
        
        # Calculate font size and position
        font_size = min(height // 2, width // (len(reading) + 1))
        try:
            font = ImageFont.truetype(self.fonts['digital'], font_size) if self.fonts['digital'] else ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        # Get text dimensions
        bbox = draw.textbbox((0, 0), reading, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Center text
        x = (width - text_width) // 2
        y = (height - text_height) // 2
        
        # Create LED glow effect
        # Outer glow
        for offset in range(3, 0, -1):
            alpha = 50 + offset * 20
            glow_color = (*config['text_color'], alpha)
            draw.text((x + offset, y + offset), reading, fill=config['text_color'], font=font)
            draw.text((x - offset, y - offset), reading, fill=config['text_color'], font=font)
        
        # Main text
        draw.text((x, y), reading, fill=config['text_color'], font=font)
        
        return image

    def create_mechanical_display(self, reading: str, width: int = 400, height: int = 150) -> Image.Image:
        """Create mechanical counter display"""
        config = self.display_configs['mechanical']
        
        # Create base image
        image = Image.new('RGB', (width, height), config['background_color'])
        draw = ImageDraw.Draw(image)
        
        # Draw outer frame
        frame_width = 5
        draw.rectangle([0, 0, width-1, height-1], 
                      outline=config['frame_color'], width=frame_width)
        
        # Draw individual digit windows
        digit_width = (width - 40) // len(reading)
        digit_height = height - 40
        
        for i, digit in enumerate(reading):
            x_start = 20 + i * digit_width
            y_start = 20
            
            # Draw digit window
            draw.rectangle([x_start, y_start, x_start + digit_width - 5, y_start + digit_height],
                          fill=(255, 255, 255), outline=(100, 100, 100), width=2)
            
            # Add mechanical separator lines
            line_spacing = digit_height // 3
            for j in range(1, 3):
                y_line = y_start + j * line_spacing
                draw.line([x_start + 5, y_line, x_start + digit_width - 10, y_line],
                         fill=(200, 200, 200), width=1)
            
            # Draw digit
            font_size = min(digit_height // 2, digit_width // 2)
            try:
                font = ImageFont.truetype(self.fonts['mechanical'], font_size) if self.fonts['mechanical'] else ImageFont.load_default()
            except:
                font = ImageFont.load_default()
            
            # Get text dimensions
            bbox = draw.textbbox((0, 0), digit, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Center digit in window
            digit_x = x_start + (digit_width - text_width) // 2
            digit_y = y_start + (digit_height - text_height) // 2
            
            # Add slight shadow for 3D effect
            draw.text((digit_x + 1, digit_y + 1), digit, fill=(180, 180, 180), font=font)
            draw.text((digit_x, digit_y), digit, fill=config['text_color'], font=font)
        
        return image

    def create_analog_dial(self, reading: str, width: int = 400, height: int = 400) -> Image.Image:
        """Create analog dial meter display"""
        config = self.display_configs['analog_dial']
        
        # Create base image
        image = Image.new('RGB', (width, height), config['background_color'])
        draw = ImageDraw.Draw(image)
        
        # Draw outer frame
        center_x, center_y = width // 2, height // 2
        outer_radius = min(width, height) // 2 - 20
        
        # Draw dial face
        draw.ellipse([center_x - outer_radius, center_y - outer_radius,
                     center_x + outer_radius, center_y + outer_radius],
                    fill=(250, 250, 250), outline=config['frame_color'], width=4)
        
        # Draw scale markings
        for i in range(12):
            angle = i * 30 * np.pi / 180  # Convert to radians
            inner_radius = outer_radius - 30
            outer_mark_radius = outer_radius - 10
            
            x1 = center_x + inner_radius * np.cos(angle)
            y1 = center_y + inner_radius * np.sin(angle)
            x2 = center_x + outer_mark_radius * np.cos(angle)
            y2 = center_y + outer_mark_radius * np.sin(angle)
            
            draw.line([x1, y1, x2, y2], fill=config['text_color'], width=3)
            
            # Draw numbers
            if i == 0:  # 12 o'clock position
                number_text = "0"
            else:
                number_text = str(i)
            
            number_radius = outer_radius - 50
            num_x = center_x + number_radius * np.cos(angle) - 5
            num_y = center_y + number_radius * np.sin(angle) - 10
            
            try:
                font = ImageFont.truetype(self.fonts['mechanical'], 20) if self.fonts['mechanical'] else ImageFont.load_default()
            except:
                font = ImageFont.load_default()
            
            draw.text((num_x, num_y), number_text, fill=config['text_color'], font=font)
        
        # Draw pointer for last digit of reading
        if reading:
            last_digit = int(reading[-1])
            pointer_angle = (last_digit * 36 - 90) * np.pi / 180  # -90 to start from top
            pointer_length = outer_radius - 40
            
            pointer_x = center_x + pointer_length * np.cos(pointer_angle)
            pointer_y = center_y + pointer_length * np.sin(pointer_angle)
            
            # Draw pointer
            draw.line([center_x, center_y, pointer_x, pointer_y], 
                     fill=(255, 0, 0), width=4)
            
            # Draw center circle
            draw.ellipse([center_x - 8, center_y - 8, center_x + 8, center_y + 8],
                        fill=(100, 100, 100), outline=(50, 50, 50), width=2)
        
        # Add reading display at bottom
        display_y = center_y + outer_radius + 30
        try:
            font = ImageFont.truetype(self.fonts['digital'], 30) if self.fonts['digital'] else ImageFont.load_default()
        except:
            font = ImageFont.load_default()
            
        bbox = draw.textbbox((0, 0), reading, font=font)
        text_width = bbox[2] - bbox[0]
        reading_x = center_x - text_width // 2
        
        draw.text((reading_x, display_y), reading, fill=config['text_color'], font=font)
        
        return image

    def add_environmental_effects(self, image: Image.Image, effect_level: float = 0.3) -> Image.Image:
        """Add realistic environmental effects to the image"""
        if random.random() > effect_level:
            return image  # Skip effects sometimes
        
        # Apply random effects
        effects = []
        
        # Lighting variations
        if random.random() < 0.4:
            enhancer = ImageEnhance.Brightness(image)
            factor = random.uniform(0.7, 1.4)
            image = enhancer.enhance(factor)
            effects.append(f"brightness_{factor:.2f}")
        
        # Contrast variations
        if random.random() < 0.3:
            enhancer = ImageEnhance.Contrast(image)
            factor = random.uniform(0.8, 1.3)
            image = enhancer.enhance(factor)
            effects.append(f"contrast_{factor:.2f}")
        
        # Slight blur (focus issues)
        if random.random() < 0.2:
            blur_radius = random.uniform(0.5, 2.0)
            image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            effects.append(f"blur_{blur_radius:.2f}")
        
        # Add noise
        if random.random() < 0.3:
            # Convert to numpy for noise addition
            img_array = np.array(image)
            noise_level = random.uniform(5, 20)
            noise = np.random.normal(0, noise_level, img_array.shape)
            noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            image = Image.fromarray(noisy_array)
            effects.append(f"noise_{noise_level:.1f}")
        
        # Perspective distortion (slight)
        if random.random() < 0.2:
            # Simple rotation
            angle = random.uniform(-3, 3)
            image = image.rotate(angle, fillcolor=image.getpixel((0, 0)))
            effects.append(f"rotation_{angle:.2f}")
        
        return image

    def generate_batch(
        self, 
        display_type: str, 
        count: int, 
        digit_count: int = 5,
        reading_type: str = "realistic"
    ) -> List[Dict[str, Any]]:
        """
        Generate a batch of simulated meter displays
        
        Args:
            display_type (str): Type of display to generate
            count (int): Number of displays to generate
            digit_count (int): Number of digits in readings
            reading_type (str): Type of reading pattern
            
        Returns:
            List of metadata for generated images
        """
        logger.info(f"Generating {count} {display_type} displays...")
        
        generated_data = []
        
        for i in range(count):
            try:
                # Generate reading
                reading = self.generate_reading(digit_count, reading_type)
                
                # Create display image based on type
                if display_type == "lcd_digital":
                    image = self.create_lcd_display(reading)
                elif display_type == "led_display":
                    image = self.create_led_display(reading)
                elif display_type == "mechanical":
                    image = self.create_mechanical_display(reading)
                elif display_type == "analog_dial":
                    image = self.create_analog_dial(reading)
                else:
                    logger.warning(f"Unknown display type: {display_type}")
                    continue
                
                # Add environmental effects
                image = self.add_environmental_effects(image)
                
                # Save image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filename = f"{display_type}_{self.session_id}_{timestamp}_{i:04d}.jpg"
                image_path = self.images_dir / filename
                
                # Convert to RGB if needed and save
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image.save(image_path, 'JPEG', quality=90)
                
                # Create metadata
                metadata = {
                    "filename": filename,
                    "reading": reading,
                    "display_type": display_type,
                    "digit_count": digit_count,
                    "reading_type": reading_type,
                    "timestamp": timestamp,
                    "session_id": self.session_id,
                    "image_size": image.size
                }
                
                # Save label file
                label_path = self.labels_dir / f"{filename.replace('.jpg', '.json')}"
                with open(label_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                generated_data.append(metadata)
                
                if (i + 1) % 50 == 0:
                    logger.info(f"Generated {i + 1}/{count} images")
                    
            except Exception as e:
                logger.error(f"Error generating image {i}: {e}")
                continue
        
        # Save batch metadata
        batch_metadata = {
            "session_id": self.session_id,
            "display_type": display_type,
            "total_generated": len(generated_data),
            "generation_time": datetime.now().isoformat(),
            "parameters": {
                "digit_count": digit_count,
                "reading_type": reading_type,
                "requested_count": count
            },
            "images": generated_data
        }
        
        metadata_file = self.metadata_dir / f"batch_{display_type}_{self.session_id}.json"
        with open(metadata_file, 'w') as f:
            json.dump(batch_metadata, f, indent=2)
        
        logger.info(f"Batch generation completed. Generated {len(generated_data)} images")
        return generated_data

    def create_realtime_simulator(self, display_type: str = "lcd_digital"):
        """Create real-time meter display simulator window"""
        logger.info(f"Starting real-time {display_type} simulator...")
        
        # Create GUI window
        root = tk.Tk()
        root.title(f"Smart Meter OCR - {display_type.replace('_', ' ').title()} Simulator")
        root.geometry("600x500")
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text=f"{display_type.replace('_', ' ').title()} Simulator",
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, pady=10)
        
        # Display canvas
        self.sim_canvas = tk.Canvas(main_frame, width=400, height=200, bg='white')
        self.sim_canvas.grid(row=1, column=0, pady=10)
        
        # Controls frame
        controls_frame = ttk.Frame(main_frame)
        controls_frame.grid(row=2, column=0, pady=10, sticky=(tk.W, tk.E))
        
        # Reading input
        ttk.Label(controls_frame, text="Reading:").grid(row=0, column=0, padx=5)
        self.reading_var = tk.StringVar(value="12345")
        reading_entry = ttk.Entry(controls_frame, textvariable=self.reading_var, width=15)
        reading_entry.grid(row=0, column=1, padx=5)
        
        # Auto-update checkbox
        self.auto_update_var = tk.BooleanVar(value=False)
        auto_check = ttk.Checkbutton(controls_frame, text="Auto Update", 
                                   variable=self.auto_update_var)
        auto_check.grid(row=0, column=2, padx=10)
        
        # Buttons
        button_frame = ttk.Frame(controls_frame)
        button_frame.grid(row=1, column=0, columnspan=3, pady=10)
        
        ttk.Button(button_frame, text="Update Display", 
                  command=lambda: self.update_simulation_display(display_type)).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Random Reading", 
                  command=self.generate_random_reading).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Image", 
                  command=lambda: self.save_simulation_image(display_type)).pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(main_frame, textvariable=self.status_var)
        status_label.grid(row=3, column=0, pady=5)
        
        # Initial display update
        self.update_simulation_display(display_type)
        
        # Auto-update loop
        def auto_update_loop():
            if self.auto_update_var.get():
                self.generate_random_reading()
                self.update_simulation_display(display_type)
            root.after(2000, auto_update_loop)  # Update every 2 seconds
        
        auto_update_loop()
        
        root.mainloop()

    def update_simulation_display(self, display_type: str):
        """Update the simulation display"""
        try:
            reading = self.reading_var.get().strip()
            if not reading or not reading.isdigit():
                self.status_var.set("Error: Reading must contain only digits")
                return
            
            # Generate display image
            if display_type == "lcd_digital":
                image = self.create_lcd_display(reading)
            elif display_type == "led_display":
                image = self.create_led_display(reading)
            elif display_type == "mechanical":
                image = self.create_mechanical_display(reading)
            elif display_type == "analog_dial":
                image = self.create_analog_dial(reading, 400, 400)
            else:
                self.status_var.set(f"Error: Unknown display type {display_type}")
                return
            
            # Convert to PhotoImage for tkinter
            from PIL import ImageTk
            photo = ImageTk.PhotoImage(image)
            
            # Update canvas
            self.sim_canvas.delete("all")
            canvas_width = self.sim_canvas.winfo_width()
            canvas_height = self.sim_canvas.winfo_height()
            
            # Center image on canvas
            x = canvas_width // 2
            y = canvas_height // 2
            self.sim_canvas.create_image(x, y, image=photo)
            self.sim_canvas.image = photo  # Keep reference
            
            self.status_var.set(f"Display updated: {reading}")
            
        except Exception as e:
            self.status_var.set(f"Error: {e}")
            logger.error(f"Error updating simulation display: {e}")

    def generate_random_reading(self):
        """Generate a random reading for simulation"""
        reading = self.generate_reading(5, "realistic")
        self.reading_var.set(reading)

    def save_simulation_image(self, display_type: str):
        """Save current simulation image"""
        try:
            reading = self.reading_var.get().strip()
            if not reading or not reading.isdigit():
                self.status_var.set("Error: Invalid reading")
                return
            
            # Generate image
            if display_type == "lcd_digital":
                image = self.create_lcd_display(reading)
            elif display_type == "led_display":
                image = self.create_led_display(reading)
            elif display_type == "mechanical":
                image = self.create_mechanical_display(reading)
            elif display_type == "analog_dial":
                image = self.create_analog_dial(reading, 400, 400)
            else:
                return
            
            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sim_{display_type}_{reading}_{timestamp}.jpg"
            filepath = self.images_dir / filename
            
            image.save(filepath, 'JPEG', quality=95)
            
            self.status_var.set(f"Image saved: {filename}")
            logger.info(f"Simulation image saved: {filepath}")
            
        except Exception as e:
            self.status_var.set(f"Error saving: {e}")
            logger.error(f"Error saving simulation image: {e}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Smart Meter Display Simulator")
    
    # Generation mode
    parser.add_argument("--type", choices=["lcd_digital", "led_display", "mechanical", "analog_dial"],
                       default="lcd_digital", help="Type of meter display to simulate")
    parser.add_argument("--count", type=int, default=100,
                       help="Number of displays to generate")
    parser.add_argument("--digit-count", type=int, default=5,
                       help="Number of digits in meter readings")
    parser.add_argument("--reading-type", choices=["sequential", "random", "realistic"],
                       default="realistic", help="Type of reading pattern")
    
    # Real-time mode
    parser.add_argument("--realtime", action="store_true",
                       help="Run real-time simulation GUI")
    
    # Output options
    parser.add_argument("--output", default="./simulated_meters",
                       help="Output directory for generated images")
    
    # Batch generation
    parser.add_argument("--batch-all", action="store_true",
                       help="Generate all display types")
    
    return parser.parse_args()


def main():
    """Main simulation function"""
    args = parse_args()
    
    try:
        # Initialize simulator
        simulator = MeterDisplaySimulator(args.output)
        
        if args.realtime:
            # Run real-time simulation
            simulator.create_realtime_simulator(args.type)
        elif args.batch_all:
            # Generate all display types
            display_types = ["lcd_digital", "led_display", "mechanical", "analog_dial"]
            for display_type in display_types:
                simulator.generate_batch(display_type, args.count // len(display_types), 
                                       args.digit_count, args.reading_type)
        else:
            # Generate specific display type
            simulator.generate_batch(args.type, args.count, 
                                   args.digit_count, args.reading_type)
        
        logger.info("Meter simulation completed successfully!")
        
    except Exception as e:
        logger.error(f"Meter simulation failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
