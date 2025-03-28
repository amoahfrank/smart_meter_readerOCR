# Smart Meter Reader OCR Tools

This directory contains various tools for developing, testing, and calibrating the Smart Meter Reader OCR system.

## Directory Structure

```
tools/
├── benchmark/             # Tools for evaluating OCR performance
│   ├── ocr_benchmark.py   # Main script for benchmarking the OCR system
│   └── requirements.txt   # Dependencies for benchmark tools
├── calibration/           # Tools for camera calibration
│   ├── calibrate_camera.py # Script for calibrating the camera module
│   └── requirements.txt   # Dependencies for calibration tools
├── data_collection/       # Tools for collecting and labeling training data
│   ├── collect_meter_data.py # Script for collecting meter readings
│   └── requirements.txt   # Dependencies for data collection tools
├── model_converter/       # Tools for converting and optimizing ML models
│   ├── convert_model.py   # Script for converting to TensorFlow Lite
│   └── requirements.txt   # Dependencies for model conversion tools
├── simulator/             # Tools for simulating meter displays
│   ├── meter_simulator.py # Script for generating simulated meter readings
│   └── requirements.txt   # Dependencies for simulator tools
├── README.md              # This file
└── requirements.txt       # Main requirements file for all tools
```

## Installation

To install all tool dependencies at once:

```bash
pip install -r tools/requirements.txt
```

For installing dependencies for a specific tool:

```bash
pip install -r tools/[tool_directory]/requirements.txt
```

## Available Tools

### Model Converter

Converts TensorFlow models to TensorFlow Lite format for deployment on the ESP32-S3.

```bash
python tools/model_converter/convert_model.py --input_model model.h5 --output_model model.tflite --quantize
```

### Camera Calibration

Helps calibrate the camera module for optimal meter reading.

```bash
python tools/calibration/calibrate_camera.py --port /dev/ttyUSB0
```

### Meter Simulator

Generates simulated meter displays for testing and training.

```bash
python tools/simulator/meter_simulator.py --meter-type lcd_digital --digit-count 5 --batch-size 10
```

### Data Collection

Collects and labels meter reading images for training.

```bash
python tools/data_collection/collect_meter_data.py --device webcam --count 50 --output-dir ./collected_data
```

For importing existing images:

```bash
python tools/data_collection/collect_meter_data.py --device import --import-path ./my_images --output-dir ./collected_data
```

### OCR Benchmark

Evaluates the performance of the OCR model on test images.

```bash
python tools/benchmark/ocr_benchmark.py --test-data ./test_images --model ./models/digit_recognition_model.tflite --visualize --save-results
```

## Development Workflow

1. **Generate Training Data**: Use the simulator to generate synthetic meter readings or collect real-world data using the data collection tool.

2. **Convert and Optimize Model**: After training your model using the project's training pipeline, use the model converter to prepare it for deployment.

3. **Calibrate Camera**: Use the calibration tool to optimize camera settings for your specific device and environment.

4. **Benchmark Performance**: Use the benchmark tool to evaluate OCR accuracy and performance.

## Contributing

When adding new tools, please follow the established patterns:
- Place tools in appropriate subdirectories
- Include a requirements.txt file for any dependencies
- Update this README with usage instructions
- Use argument parsing for command line options
- Include appropriate error handling and logging

## License

All tools are part of the Smart Meter Reader OCR project and are licensed under the MIT License.
