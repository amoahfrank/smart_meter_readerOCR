# Smart Meter Reader OCR

A proof-of-concept device for reading and transmitting numeric values from various physical meters using advanced OCR and AI/ML technologies.

## Project Overview

The Smart Meter Reader OCR is an intelligent, adaptable device designed to be placed over existing utility meters (water, gas, electricity) to automate the reading process. The device captures images of various meter types (rotary dials, digital LCD, LED displays), processes them locally using OCR with TensorFlow Lite, displays the reading on an integrated e-paper display, and wirelessly transmits the data to a cloud endpoint.

### Key Features

- **Camera-based Meter Reading**: Automatically captures images of meter displays
- **Local OCR Processing**: Efficiently extracts numeric values using TensorFlow Lite
- **AI-Driven Recognition**: Adapts to different meter types and lighting conditions
- **Wireless Connectivity**: Transmits data via WiFi, BLE, or optional LoRaWAN
- **Energy Efficient**: Battery-powered with deep sleep modes for extended operation
- **Secure Communication**: Industry-standard TLS/SSL implementation
- **Over-the-Air Updates**: Secure OTA firmware updates
- **Easy Installation**: Simple mounting mechanism for existing meters

## Architecture

The system architecture follows a modular design with these key components:

1. **Camera Module**: Handles image acquisition and basic processing
2. **OCR Engine**: Preprocesses images and runs the TensorFlow Lite model
3. **Display Module**: Manages the e-paper display and user interface
4. **Connectivity Module**: Handles wireless communications (WiFi/BLE/LoRaWAN)
5. **Power Management**: Optimizes battery usage and implements sleep modes
6. **Security Module**: Provides encryption, secure storage, and OTA functionality
7. **Configuration System**: Manages device settings and calibration

## Hardware Requirements

### Bill of Materials (BOM)

| Component | Recommended Part | Description |
|-----------|------------------|-------------|
| Microcontroller | ESP32-S3-WROOM-1 | Main processor with AI acceleration |
| Camera | OV2640 | 2MP camera module with SCCB interface |
| Display | 1.54" e-Paper Module | Low power E-Ink display |
| Battery | 3.7V 2000mAh LiPo | Rechargeable battery |
| Optional Solar | 5V 1W Solar Panel | Solar charging capability |
| Wireless (Optional) | SX1262 LoRa Module | Long-range, low-power communication |
| Buttons | 2x Tactile Switches | User interface |
| LEDs | 2x Status LEDs | Visual indicators |
| PCB | Custom PCB | Main circuit board |
| Enclosure | Custom 3D-printed case | Weather-resistant housing |
| Misc | Resistors, capacitors, etc. | Support components |

## Software Dependencies

- **ESP-IDF v5.0+**: Espressif IoT Development Framework
- **TensorFlow Lite for Microcontrollers**: For running the OCR model
- **Espressif ESP32 Camera Driver**: For interfacing with the OV2640 camera
- **GoodDisplay e-Paper Library**: For controlling the e-paper display
- **Arduino LoRaWAN Library** (optional): For LoRaWAN connectivity
- **MbedTLS**: For TLS/SSL security implementation

## Directory Structure

```
smart_meter_readerOCR/
├── .github/
│   └── workflows/
│       └── ci.yml
├── components/
│   ├── camera/
│   │   ├── include/
│   │   │   └── camera.h
│   │   ├── camera.c
│   │   └── CMakeLists.txt
│   ├── display/
│   │   ├── include/
│   │   │   └── display.h
│   │   ├── display.c
│   │   └── CMakeLists.txt
│   ├── ocr/
│   │   ├── include/
│   │   │   ├── ocr.h
│   │   │   ├── image_processing.h
│   │   │   └── model_interface.h
│   │   ├── ocr.c
│   │   ├── image_processing.c
│   │   ├── model_interface.c
│   │   └── CMakeLists.txt
│   ├── connectivity/
│   │   ├── include/
│   │   │   ├── wifi_manager.h
│   │   │   ├── ble_manager.h
│   │   │   └── lora_manager.h
│   │   ├── wifi_manager.c
│   │   ├── ble_manager.c
│   │   ├── lora_manager.c
│   │   └── CMakeLists.txt
│   ├── power_mgmt/
│   │   ├── include/
│   │   │   └── power_mgmt.h
│   │   ├── power_mgmt.c
│   │   └── CMakeLists.txt
│   ├── security/
│   │   ├── include/
│   │   │   ├── security_manager.h
│   │   │   └── ota_manager.h
│   │   ├── security_manager.c
│   │   ├── ota_manager.c
│   │   └── CMakeLists.txt
│   └── configuration/
│       ├── include/
│       │   └── configuration.h
│       ├── configuration.c
│       └── CMakeLists.txt
├── main/
│   ├── include/
│   │   ├── app_main.h
│   │   └── state_machine.h
│   ├── app_main.c
│   ├── state_machine.c
│   └── CMakeLists.txt
├── models/
│   ├── digit_recognition_model.tflite
│   └── README.md
├── tools/
│   ├── model_converter/
│   │   ├── convert_model.py
│   │   └── requirements.txt
│   └── calibration/
│       ├── calibrate_camera.py
│       └── requirements.txt
├── docs/
│   ├── hardware_setup.md
│   ├── software_guide.md
│   ├── ocr_model.md
│   ├── security.md
│   └── images/
├── partitions.csv
├── sdkconfig.defaults
├── CMakeLists.txt
├── LICENSE
└── README.md
```

## Build and Installation

### Prerequisites

1. ESP-IDF v5.0+ installed and properly configured
2. Python 3.7+ for the development tools
3. Required Python packages: `pip install -r tools/requirements.txt`

### Building the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/amoahfrank/smart_meter_readerOCR.git
   cd smart_meter_readerOCR
   ```

2. Configure the project:
   ```bash
   idf.py menuconfig
   ```
   - Configure WiFi credentials under "Smart Meter Reader Configuration"
   - Adjust camera settings if necessary
   - Configure OCR model parameters

3. Build and flash:
   ```bash
   idf.py build
   idf.py -p (PORT) flash
   ```

4. Monitor output (optional):
   ```bash
   idf.py -p (PORT) monitor
   ```

## Security Implementation

The device implements multiple layers of security:

- **Secure Boot**: Verifies firmware integrity at startup
- **Encrypted Storage**: Securely stores WiFi credentials and configuration
- **TLS/SSL**: Encrypts all data transmitted over WiFi
- **Secure BLE**: Implements encryption and authentication for BLE communications
- **Secure OTA**: Verifies authenticity of firmware updates
- **Physical Security**: Detects and reports tampering attempts

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
