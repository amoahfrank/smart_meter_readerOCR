# Smart Meter Reader OCR - Quick Start Guide

**Get up and running with Smart Meter OCR in 30 minutes!**

This guide will walk you through setting up the complete Smart Meter Reader OCR system, from data collection to deployment on ESP32.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Quick Demo](#quick-demo)
4. [Collect Training Data](#collect-training-data)
5. [Train Your Model](#train-your-model)
6. [Test and Evaluate](#test-and-evaluate)
7. [Deploy to ESP32](#deploy-to-esp32)
8. [Next Steps](#next-steps)

---

## Prerequisites

### Hardware Requirements
- **Computer**: Windows, macOS, or Linux with Python 3.7+
- **Webcam**: Any USB camera (built-in laptop camera works)
- **ESP32 Board** (optional): ESP32-S3 with camera module for deployment

### Software Requirements
- **Python 3.7+**: [Download here](https://python.org/downloads)
- **Git**: [Download here](https://git-scm.com/downloads)
- **ESP-IDF** (optional, for ESP32): [Setup guide](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/get-started/)

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/amoahfrank/smart_meter_readerOCR.git
cd smart_meter_readerOCR
```

### Step 2: Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\\Scripts\\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r tools/requirements.txt
```

### Step 3: Verify Installation

```bash
# Test data collection tool
python tools/data_collection/collect_meter_data.py --help

# Test simulator
python tools/simulator/meter_simulator.py --help

# Test training pipeline
python tools/model_converter/train_model.py --help
```

If all commands run without errors, you're ready to go! 🎉

---

## Quick Demo

Let's start with a quick demo using the built-in simulator:

### Generate Sample Data

```bash
# Generate 100 synthetic meter displays
python tools/simulator/meter_simulator.py \
    --type lcd_digital \
    --count 100 \
    --output ./demo_data
```

**What this does:**
- Creates realistic LCD meter displays
- Generates images with random readings
- Saves images and labels for training

### Real-time Simulator

```bash
# Launch interactive simulator
python tools/simulator/meter_simulator.py --realtime --type lcd_digital
```

**Try this:**
- Enter different readings (e.g., "12345", "98765")
- Click "Random Reading" to generate new readings
- Save images you like for training data

---

## Collect Training Data

Now let's collect real training data using your webcam:

### Method 1: Interactive Collection

```bash
# Start webcam data collection
python tools/data_collection/collect_meter_data.py \
    --source webcam \
    --count 50 \
    --output ./my_training_data
```

**Collection Tips:**
- Use a calculator or digital clock as a "practice meter"
- Try different angles and lighting conditions
- Press SPACE to capture, enter the correct reading
- Press Q to quit when done

### Method 2: Process Existing Images

```bash
# If you have meter photos already
python tools/data_collection/collect_meter_data.py \
    --source import \
    --input-dir ./my_meter_photos \
    --output ./my_training_data
```

### Method 3: Generate Synthetic Data

```bash
# Generate large synthetic dataset
python tools/data_collection/collect_meter_data.py \
    --source synthetic \
    --count 1000 \
    --output ./my_training_data
```

### Export for Training

```bash
# Prepare data for training
python tools/data_collection/collect_meter_data.py \
    --export \
    --output ./my_training_data
```

**Expected result:** You should now have directories with images, labels, and training splits.

---

## Train Your Model

### Basic Training (Recommended for Beginners)

```bash
# Train a simple CNN model
python tools/model_converter/train_model.py \
    --data-dir ./my_training_data \
    --model-type cnn \
    --epochs 20 \
    --batch-size 16 \
    --output-dir ./my_models
```

**What to expect:**
- Training will take 10-30 minutes depending on your computer
- You'll see accuracy improving each epoch
- Final model will be saved automatically

**Example output:**
```
Epoch 1/20
32/32 [==============================] - 12s - loss: 2.1234 - accuracy: 0.2500
Epoch 2/20
32/32 [==============================] - 8s - loss: 1.5432 - accuracy: 0.4500
...
Epoch 20/20
32/32 [==============================] - 8s - loss: 0.1234 - accuracy: 0.9500

Training completed! Model saved to: ./my_models/models/digit_recognition_model.tflite
```

### Advanced Training (For Better Accuracy)

```bash
# Train advanced ResNet model
python tools/model_converter/train_model.py \
    --data-dir ./my_training_data \
    --model-type resnet \
    --epochs 50 \
    --batch-size 32 \
    --use-augmentation \
    --output-dir ./my_models
```

---

## Test and Evaluate

### Benchmark Your Model

```bash
# Comprehensive model evaluation
python tools/benchmark/ocr_benchmark.py \
    --model ./my_models/models/digit_recognition_model.tflite \
    --test-data ./my_training_data/training_export/test.json \
    --full-report \
    --visualize
```

**Results you'll get:**
- Overall accuracy score
- Confusion matrix showing common mistakes
- Performance charts and graphs
- Recommendations for improvement

### Compare Multiple Models

```bash
# Compare different model architectures
python tools/benchmark/ocr_benchmark.py \
    --compare-models ./my_models/models/*.tflite \
    --test-data ./my_training_data/training_export/test.json \
    --full-report
```

### Check Results

Look in `./benchmark_results/` for:
- `benchmark_report_XXXXXX.md` - Human-readable report
- `plots/` - Accuracy charts and confusion matrices
- `reports/` - Detailed JSON results

---

## Deploy to ESP32

### Generate ESP32 Configuration

```bash
# Create optimized camera settings
python tools/calibration/calibrate_camera.py \
    --generate-esp32-config \
    --config-output camera_config.h
```

### Build ESP32 Firmware

```bash
# Set up ESP-IDF environment (one-time setup)
. $HOME/esp/esp-idf/export.sh

# Configure the project
idf.py menuconfig
# Navigate to "Smart Meter Reader Configuration"
# Set WiFi credentials and other settings

# Copy your trained model
cp ./my_models/models/digit_recognition_model.tflite ./main/

# Build and flash
idf.py build
idf.py -p /dev/ttyUSB0 flash monitor
```

### Test on Device

1. **Power on** your ESP32 with camera module
2. **Position** a meter display in front of the camera
3. **Check serial output** for recognition results
4. **Verify readings** against actual meter values

**Expected serial output:**
```
I (12345) main: Smart Meter Reader OCR Started
I (12346) camera: Camera initialized successfully
I (12347) wifi: Connected to WiFi network
I (15000) ocr: Processing image...
I (15500) ocr: Recognized reading: 12345 (confidence: 0.95)
I (15501) mqtt: Published reading to cloud
```

---

## Next Steps

Congratulations! You now have a working Smart Meter Reader OCR system. Here's what you can do next:

### Improve Accuracy
1. **Collect more diverse data** in different lighting conditions
2. **Fine-tune your model** with real meter images
3. **Try advanced architectures** like EfficientNet
4. **Use ensemble methods** combining multiple models

### Enhance Functionality
1. **Add wireless communication** (WiFi, LoRaWAN, cellular)
2. **Implement cloud integration** with AWS IoT or Google Cloud
3. **Create mobile app** for configuration and monitoring
4. **Add support for analog meters** with pointer detection

### Scale Deployment
1. **Design custom PCB** for production units
2. **Add solar charging** for remote installations
3. **Implement mesh networking** for multi-meter systems
4. **Create management dashboard** for fleet monitoring

### Advanced Features
1. **Time series analysis** for usage pattern detection
2. **Anomaly detection** for leak/theft identification
3. **Predictive maintenance** alerts
4. **Integration with smart home systems**

---

## Troubleshooting

### Common Issues

**"No cameras found" during data collection**
```bash
# List available cameras
python -c "import cv2; print([i for i in range(5) if cv2.VideoCapture(i).isOpened()])"

# Try different camera ID
python tools/data_collection/collect_meter_data.py --source webcam --camera-id 1
```

**Training is very slow**
```bash
# Use smaller batch size and fewer epochs for testing
python tools/model_converter/train_model.py \
    --data-dir ./my_training_data \
    --batch-size 8 \
    --epochs 10
```

**Model accuracy is poor**
- Collect more diverse training data (different lighting, angles)
- Check that your labels are correct
- Try data augmentation: `--use-augmentation`
- Use a more complex model: `--model-type resnet`

**ESP32 out of memory**
- Use a smaller model or increase memory allocation
- Check `kTensorArenaSize` in the ESP32 code
- Consider streaming processing for large images

### Getting Help

- **Documentation**: Check the comprehensive [AI/ML Training Guide](docs/AI_ML_Training_Guide.md)
- **Issues**: Open an issue on GitHub with details about your problem
- **Discussions**: Join the project discussions for community support

---

## Success Stories

**"Great educational project! My students learned about computer vision, IoT, and embedded systems all in one project."** - University Professor

---

## What's Next?

You've successfully built a complete Smart Meter Reader OCR system! This is just the beginning. The computer vision and AI techniques you've learned can be applied to many other applications:

- License plate recognition
- Document digitization
- Quality control in manufacturing
- Inventory management
- Medical image analysis

Keep experimenting, learning, and building amazing AI-powered solutions! 🚀

---

*For the complete technical documentation, see the [AI/ML Training Guide](docs/AI_ML_Training_Guide.md). For hardware setup and deployment details, check the [Hardware Setup Guide](docs/hardware_setup.md).*
