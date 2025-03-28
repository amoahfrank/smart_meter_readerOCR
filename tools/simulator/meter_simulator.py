#!/usr/bin/env python3
"""
Meter Display Simulator for Smart Meter Reader OCR project.

This tool generates realistic simulations of various utility meter displays
for testing and development purposes. It can create rotary dials, LCD, and LED
display types with configurable properties like noise, lighting conditions, etc.
"""

import os
import sys
import argparse
import cv2
import numpy as np
import random
import json
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# Default settings
DEFAULT_SETTINGS = {
    "meter_type": "lcd_digital",  # lcd_digital, led_digital, rotary_dial
    "digit_count": 5,
    "background_color": (245, 245, 245),
    "digit_color": (10, 10, 10),
    "noise_level": 0.05,
    "blur_level": 0,
    "light_condition": "normal",  # normal, dim, bright, glare
    "rotation_angle": 0,
    "perspective_distortion": 0,
    "width": 640,
    "height": 480,
    "output_format": "jpg"
}

# Font settings
FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",  # Linux
    "/Library/Fonts/Courier New.ttf",  # macOS
    "C:\\Windows\\Fonts\\cour.ttf",  # Windows
    "DejaVuSansMono.ttf"  # Relative path
]

class MeterSimulator:
    """Generate simulated meter display images."""
    
    def __init__(self, settings=None):
        """Initialize simulator with given settings."""
        self.settings = DEFAULT_SETTINGS.copy()
        if settings:
            self.settings.update(settings)
        
        # Load font for digital displays
        self.font = self._load_font()
        
        # Seed random number generator for reproducibility if seed is provided
        if "random_seed" in self.settings:
            random.seed(self.settings["random_seed"])
            np.random.seed(self.settings["random_seed"])
    
    def _load_font(self):
        """Load a monospace font for digital displays."""
        font_size = int(min(self.settings["width"], self.settings["height"]) / (self.settings["digit_count"] * 2))
        
        for font_path in FONT_PATHS:
            try:
                return ImageFont.truetype(font_path, font_size)
            except (OSError, IOError):
                continue
        
        # Fallback to default font
        print("Warning: Could not load monospace font, using default")
        return ImageFont.load_default()
    
    def generate_image(self, meter_value=None):
        """Generate a simulated meter image."""
        if meter_value is None:
            # Generate random meter value if not provided
            max_value = 10 ** self.settings["digit_count"] - 1
            meter_value = random.randint(0, max_value)
        
        # Format the value as a string with leading zeros
        formatted_value = str(meter_value).zfill(self.settings["digit_count"])
        
        # Create base image based on meter type
        if self.settings["meter_type"] == "lcd_digital":
            image = self._create_lcd_display(formatted_value)
        elif self.settings["meter_type"] == "led_digital":
            image = self._create_led_display(formatted_value)
        elif self.settings["meter_type"] == "rotary_dial":
            image = self._create_rotary_display(formatted_value)
        else:
            raise ValueError(f"Unknown meter type: {self.settings['meter_type']}")
        
        # Apply post-processing effects
        image = self._apply_effects(image)
        
        return image, formatted_value
    
    def _create_lcd_display(self, value):
        """Create an LCD-style digital display image."""
        # Create base image with background color
        img = Image.new('RGB', (self.settings["width"], self.settings["height"]), 
                      self.settings["background_color"])
        draw = ImageDraw.Draw(img)
        
        # Calculate positions for digits
        digit_width = self.font.getbbox("0")[2]
        total_width = digit_width * len(value)
        start_x = (self.settings["width"] - total_width) // 2
        start_y = (self.settings["height"] - self.font.getbbox("0")[3]) // 2
        
        # Draw each digit
        for i, digit in enumerate(value):
            position = (start_x + i * digit_width, start_y)
            draw.text(position, digit, font=self.font, fill=self.settings["digit_color"])
        
        # Draw LCD frame
        frame_padding = 20
        frame_x0 = start_x - frame_padding
        frame_y0 = start_y - frame_padding
        frame_x1 = start_x + total_width + frame_padding
        frame_y1 = start_y + self.font.getbbox("0")[3] + frame_padding
        
        draw.rectangle([frame_x0, frame_y0, frame_x1, frame_y1], 
                      outline=(100, 100, 100), width=2)
        
        return img
    
    def _create_led_display(self, value):
        """Create an LED-style digital display image."""
        # Similar to LCD but with different styling
        img = Image.new('RGB', (self.settings["width"], self.settings["height"]), 
                      (0, 0, 0))  # Black background
        draw = ImageDraw.Draw(img)
        
        # Calculate positions for digits
        digit_width = self.font.getbbox("0")[2]
        total_width = digit_width * len(value)
        start_x = (self.settings["width"] - total_width) // 2
        start_y = (self.settings["height"] - self.font.getbbox("0")[3]) // 2
        
        # Draw each digit in red LED color
        for i, digit in enumerate(value):
            position = (start_x + i * digit_width, start_y)
            draw.text(position, digit, font=self.font, fill=(255, 50, 50))  # Red LED color
        
        # Add LED glow effect
        img = img.filter(ImageFilter.GaussianBlur(radius=1))
        
        return img
    
    def _create_rotary_display(self, value):
        """Create a mechanical rotary dial display image."""
        # Create base image
        img = Image.new('RGB', (self.settings["width"], self.settings["height"]), 
                      (220, 220, 220))  # Light gray background
        draw = ImageDraw.Draw(img)
        
        # Calculate dial sizes and positions
        dial_radius = min(self.settings["width"], self.settings["height"]) // (2 * self.settings["digit_count"] + 1)
        dial_spacing = dial_radius * 2.5
        total_width = dial_spacing * len(value)
        start_x = (self.settings["width"] - total_width) // 2 + dial_radius
        start_y = self.settings["height"] // 2
        
        # Draw each dial
        for i, digit in enumerate(value):
            digit_int = int(digit)
            center_x = start_x + i * dial_spacing
            center_y = start_y
            
            # Draw dial background
            draw.ellipse([center_x - dial_radius, center_y - dial_radius,
                         center_x + dial_radius, center_y + dial_radius],
                        fill=(255, 255, 255), outline=(100, 100, 100))
            
            # Draw tick marks
            for j in range(10):
                angle = j * 36 - 90  # -90 degrees is the top
                outer_x = center_x + int(0.8 * dial_radius * np.cos(np.radians(angle)))
                outer_y = center_y + int(0.8 * dial_radius * np.sin(np.radians(angle)))
                inner_x = center_x + int(0.7 * dial_radius * np.cos(np.radians(angle)))
                inner_y = center_y + int(0.7 * dial_radius * np.sin(np.radians(angle)))
                draw.line([inner_x, inner_y, outer_x, outer_y], fill=(0, 0, 0), width=2)
                
                # Draw numbers
                text_x = center_x + int(0.6 * dial_radius * np.cos(np.radians(angle)))
                text_y = center_y + int(0.6 * dial_radius * np.sin(np.radians(angle)))
                # Adjust for text centering
                text_x -= 5
                text_y -= 5
                draw.text((text_x, text_y), str(j), fill=(0, 0, 0))
            
            # Draw pointer
            angle = digit_int * 36 - 90
            pointer_x = center_x + int(0.7 * dial_radius * np.cos(np.radians(angle)))
            pointer_y = center_y + int(0.7 * dial_radius * np.sin(np.radians(angle)))
            draw.line([center_x, center_y, pointer_x, pointer_y], fill=(255, 0, 0), width=3)
            
            # Draw center cap
            draw.ellipse([center_x - dial_radius/6, center_y - dial_radius/6,
                         center_x + dial_radius/6, center_y + dial_radius/6],
                        fill=(150, 150, 150))
        
        return img
    
    def _apply_effects(self, image):
        """Apply post-processing effects to the image."""
        # Convert PIL Image to numpy array for OpenCV operations
        img_array = np.array(image)
        
        # Apply rotation if specified
        if self.settings["rotation_angle"] != 0:
            center = (img_array.shape[1] // 2, img_array.shape[0] // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, self.settings["rotation_angle"], 1.0)
            img_array = cv2.warpAffine(img_array, rotation_matrix, (img_array.shape[1], img_array.shape[0]))
        
        # Apply perspective distortion if specified
        if self.settings["perspective_distortion"] > 0:
            h, w = img_array.shape[:2]
            distortion = self.settings["perspective_distortion"] * w // 10
            
            # Define source points (original image corners)
            src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            
            # Define destination points (distorted image corners)
            dst_pts = np.float32([
                [distortion, distortion],
                [w - distortion, distortion],
                [w, h],
                [0, h]
            ])
            
            # Apply perspective transformation
            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            img_array = cv2.warpPerspective(img_array, matrix, (w, h))
        
        # Apply lighting conditions
        if self.settings["light_condition"] == "dim":
            img_array = cv2.convertScaleAbs(img_array, alpha=0.7, beta=-30)
        elif self.settings["light_condition"] == "bright":
            img_array = cv2.convertScaleAbs(img_array, alpha=1.2, beta=30)
        elif self.settings["light_condition"] == "glare":
            # Add a glare spot
            glare_x = random.randint(0, img_array.shape[1] - 1)
            glare_y = random.randint(0, img_array.shape[0] - 1)
            glare_size = min(img_array.shape[0], img_array.shape[1]) // 4
            
            # Create glare mask
            mask = np.zeros_like(img_array)
            cv2.circle(mask, (glare_x, glare_y), glare_size, (255, 255, 255), -1)
            mask = cv2.GaussianBlur(mask, (glare_size * 2 + 1, glare_size * 2 + 1), 0)
            
            # Apply glare
            img_array = cv2.addWeighted(img_array, 1, mask, 0.5, 0)
        
        # Apply blur if specified
        if self.settings["blur_level"] > 0:
            blur_size = 2 * int(self.settings["blur_level"] * 5) + 1
            img_array = cv2.GaussianBlur(img_array, (blur_size, blur_size), 0)
        
        # Apply noise if specified
        if self.settings["noise_level"] > 0:
            noise = np.random.normal(0, self.settings["noise_level"] * 255, img_array.shape).astype(np.uint8)
            img_array = cv2.add(img_array, noise)
        
        # Convert back to PIL Image
        return Image.fromarray(img_array)
    
    def save_image(self, image, value, output_dir, prefix="meter"):
        """Save generated image to file."""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{self.settings['meter_type']}_{value}_{timestamp}.{self.settings['output_format']}"
        filepath = os.path.join(output_dir, filename)
        
        # Save image
        image.save(filepath)
        print(f"Image saved to {filepath}")
        
        # Save metadata
        metadata_file = f"{os.path.splitext(filepath)[0]}.json"
        with open(metadata_file, 'w') as f:
            metadata = self.settings.copy()
            metadata["value"] = value
            metadata["timestamp"] = timestamp
            json.dump(metadata, f, indent=4)
        
        return filepath

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate simulated meter display images')
    parser.add_argument('--meter-type', choices=['lcd_digital', 'led_digital', 'rotary_dial'], 
                        default='lcd_digital', help='Type of meter display to simulate')
    parser.add_argument('--value', type=int, help='Specific value to display (random if not provided)')
    parser.add_argument('--digit-count', type=int, default=5, help='Number of digits in the display')
    parser.add_argument('--noise', type=float, default=0.05, help='Noise level (0.0 to 1.0)')
    parser.add_argument('--blur', type=float, default=0, help='Blur level (0.0 to 1.0)')
    parser.add_argument('--light', choices=['normal', 'dim', 'bright', 'glare'], 
                        default='normal', help='Lighting condition')
    parser.add_argument('--rotation', type=float, default=0, help='Rotation angle in degrees')
    parser.add_argument('--perspective', type=float, default=0, 
                        help='Perspective distortion level (0.0 to 1.0)')
    parser.add_argument('--width', type=int, default=640, help='Image width in pixels')
    parser.add_argument('--height', type=int, default=480, help='Image height in pixels')
    parser.add_argument('--batch-size', type=int, default=1, help='Number of images to generate')
    parser.add_argument('--output-dir', default='./generated_meters', help='Output directory')
    parser.add_argument('--output-format', choices=['jpg', 'png'], default='jpg', help='Output file format')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--config', help='Path to JSON configuration file')
    return parser.parse_args()

def main():
    """Main entry point of the script."""
    args = parse_args()
    
    # Load configuration from file if provided
    settings = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            settings = json.load(f)
    
    # Override with command line arguments
    if args.meter_type:
        settings["meter_type"] = args.meter_type
    if args.digit_count:
        settings["digit_count"] = args.digit_count
    if args.noise is not None:
        settings["noise_level"] = args.noise
    if args.blur is not None:
        settings["blur_level"] = args.blur
    if args.light:
        settings["light_condition"] = args.light
    if args.rotation is not None:
        settings["rotation_angle"] = args.rotation
    if args.perspective is not None:
        settings["perspective_distortion"] = args.perspective
    if args.width:
        settings["width"] = args.width
    if args.height:
        settings["height"] = args.height
    if args.output_format:
        settings["output_format"] = args.output_format
    if args.seed:
        settings["random_seed"] = args.seed
    
    # Create simulator
    simulator = MeterSimulator(settings)
    
    # Generate images
    for i in range(args.batch_size):
        image, value = simulator.generate_image(args.value)
        simulator.save_image(image, value, args.output_dir)
    
    print(f"Generated {args.batch_size} simulated meter images")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSimulation aborted by user")
        sys.exit(0)
