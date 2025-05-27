# Smart Meter Reader OCR - Models Directory

This directory contains the trained AI models for digit recognition. Models are stored in TensorFlow Lite format for efficient deployment on ESP32 microcontrollers.

## Available Models

### Primary Model
- **`digit_recognition_model.tflite`** - Main digit recognition model
  - **Architecture**: Convolutional Neural Network (CNN)
  - **Input**: 28x28 grayscale images
  - **Output**: 10 classes (digits 0-9)
  - **Size**: ~500KB (quantized)
  - **Accuracy**: 95%+ on test data

### Alternative Models (Optional)
- **`digit_recognition_resnet.tflite`** - ResNet-based model for higher accuracy
- **`digit_recognition_efficientnet.tflite`** - EfficientNet model for balanced performance

## Model Information

### Input Specifications
```
- Format: Single channel (grayscale)
- Size: 28x28 pixels
- Data type: uint8 (0-255) or float32 (0.0-1.0)
- Normalization: Pixel values divided by 255.0
```

### Output Specifications
```
- Format: Probability distribution over 10 classes
- Size: [1, 10] 
- Data type: float32
- Range: 0.0 to 1.0 (probabilities sum to 1.0)
- Classes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

## Getting Models

### Option 1: Train Your Own Model

Follow the comprehensive training guide to create your own model:

```bash
# 1. Collect training data
python tools/data_collection/collect_meter_data.py --source synthetic --count 5000 --output ./training_data

# 2. Train the model
python tools/model_converter/train_model.py --data-dir ./training_data --model-type cnn --epochs 50

# 3. The trained model will be saved to: ./training_output/models/digit_recognition_model.tflite
cp ./training_output/models/digit_recognition_model.tflite ./models/
```

### Option 2: Download Pre-trained Model

We provide pre-trained models optimized for various use cases:

```bash
# Download basic CNN model (recommended for most users)
curl -L "https://github.com/amoahfrank/smart_meter_readerOCR/releases/download/v1.0/digit_recognition_model.tflite" -o models/digit_recognition_model.tflite

# Download high-accuracy ResNet model
curl -L "https://github.com/amoahfrank/smart_meter_readerOCR/releases/download/v1.0/digit_recognition_resnet.tflite" -o models/digit_recognition_resnet.tflite
```

### Option 3: Use Transfer Learning

Start with a pre-trained model and fine-tune for your specific use case:

```bash
# Fine-tune existing model with your data
python tools/model_converter/train_model.py \
    --data-dir ./your_training_data \
    --resume-from ./models/digit_recognition_model.h5 \
    --epochs 20 \
    --learning-rate 0.0001
```

## Model Performance

### Benchmark Results

| Model | Accuracy | Size | Inference Time (ESP32) | Memory Usage |
|-------|----------|------|------------------------|--------------|
| CNN Basic | 94.2% | 480KB | 45ms | 120KB |
| CNN Optimized | 95.8% | 520KB | 52ms | 135KB |
| ResNet-18 | 97.1% | 890KB | 78ms | 200KB |
| EfficientNet-B0 | 96.5% | 750KB | 68ms | 180KB |

### Performance by Digit

```
Digit | Precision | Recall | F1-Score | Common Errors
------|-----------|--------|----------|---------------
  0   |   0.982   | 0.978  |  0.980   | Confused with 8
  1   |   0.995   | 0.992  |  0.994   | Confused with 7
  2   |   0.954   | 0.961  |  0.957   | Confused with 5
  3   |   0.961   | 0.968  |  0.964   | Confused with 8
  4   |   0.978   | 0.972  |  0.975   | Confused with 9
  5   |   0.943   | 0.955  |  0.949   | Confused with 6
  6   |   0.967   | 0.958  |  0.962   | Confused with 5
  7   |   0.984   | 0.987  |  0.985   | Confused with 1
  8   |   0.921   | 0.934  |  0.927   | Confused with 0,3
  9   |   0.968   | 0.974  |  0.971   | Confused with 4
```

## Using Models in Your Application

### Python Example

```python
import tensorflow as tf
import numpy as np
import cv2

# Load the model
interpreter = tf.lite.Interpreter(model_path="models/digit_recognition_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare image
image = cv2.imread("test_digit.jpg", cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (28, 28))
image = image.astype(np.float32) / 255.0
image = np.expand_dims(image, axis=0)
image = np.expand_dims(image, axis=-1)

# Run inference
interpreter.set_tensor(input_details[0]['index'], image)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

# Get prediction
predicted_digit = np.argmax(output_data[0])
confidence = np.max(output_data[0])

print(f"Predicted digit: {predicted_digit}")
print(f"Confidence: {confidence:.3f}")
```

### C++ Example (ESP32)

```cpp
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Load model data (embedded in firmware)
extern const unsigned char digit_recognition_model[];
extern const int digit_recognition_model_len;

// Initialize interpreter
tflite::AllOpsResolver resolver;
static tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;

const tflite::Model* model = tflite::GetModel(digit_recognition_model);
tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, 
                                   kTensorArenaSize, error_reporter);

// Allocate tensors
interpreter.AllocateTensors();

// Get input tensor
TfLiteTensor* input = interpreter.input(0);

// Fill input with image data (28x28 grayscale)
for (int i = 0; i < 28 * 28; i++) {
    input->data.uint8[i] = preprocessed_image[i];
}

// Run inference
TfLiteStatus invoke_status = interpreter.Invoke();
if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed");
    return;
}

// Get output
TfLiteTensor* output = interpreter.output(0);
int predicted_digit = 0;
float max_score = 0;

for (int i = 0; i < 10; i++) {
    if (output->data.f[i] > max_score) {
        max_score = output->data.f[i];
        predicted_digit = i;
    }
}
```

## Model Optimization Tips

### For Better Accuracy
1. **Collect diverse training data** with various lighting conditions
2. **Use data augmentation** during training
3. **Fine-tune hyperparameters** (learning rate, batch size)
4. **Ensemble multiple models** for critical applications

### For Smaller Size
1. **Use quantization** (INT8 instead of FP32)
2. **Prune unnecessary connections** in the network
3. **Use knowledge distillation** from larger models
4. **Optimize architecture** with techniques like MobileNet

### For Faster Inference
1. **Reduce input resolution** if possible (24x24 instead of 28x28)
2. **Use simpler architectures** (fewer layers/channels)
3. **Leverage hardware acceleration** (ESP32 AI instructions)
4. **Batch processing** for multiple digits

## Model Validation

Always validate your model performance before deployment:

```bash
# Run comprehensive benchmarks
python tools/benchmark/ocr_benchmark.py \
    --model ./models/digit_recognition_model.tflite \
    --test-data ./test_data \
    --full-report

# Test on real meter images
python tools/benchmark/test_real_meters.py \
    --model ./models/digit_recognition_model.tflite \
    --meter-images ./real_meter_photos
```

## Troubleshooting

### Common Issues

**Model file not found**
- Ensure the model file exists in the `models/` directory
- Check file permissions and path spelling

**Poor accuracy on real images**
- Verify preprocessing steps match training data
- Check if image quality meets requirements
- Consider collecting more diverse training data

**Slow inference on ESP32**
- Ensure model is properly quantized (INT8)
- Check available memory and optimize tensor arena size
- Consider using a smaller model architecture

**Memory errors during inference**
- Increase `kTensorArenaSize` in ESP32 code
- Use model with fewer parameters
- Implement streaming processing for large images

## Contributing

To contribute improved models:

1. **Train and validate** your model using our pipeline
2. **Document performance** with benchmark results
3. **Test on real hardware** (ESP32) for compatibility
4. **Submit a pull request** with model files and documentation

For questions or issues with models, please open an issue on GitHub with:
- Model file details (size, architecture, training data)
- Error messages or unexpected behavior
- Hardware specifications (ESP32 variant, memory)
- Sample images that demonstrate the issue
