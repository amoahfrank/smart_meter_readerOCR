/**
 * @file ocr.c
 * @brief OCR implementation for Smart Meter Reader OCR
 * 
 * This file implements the OCR functionality using TensorFlow Lite for
 * Microcontrollers. It handles preprocessing of images and recognition
 * of meter readings.
 * 
 * @author Frank Amoah A.K.A SiRWaTT Smart Meter Reader OCR Team Lead
 * @date 2023
 */

#include <stdio.h>
#include <string.h>
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_camera.h"
#include "esp_heap_caps.h"

// TensorFlow Lite includes
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Include component headers
#include "image_processing.h"
#include "model_interface.h"

// Include local headers
#include "ocr.h"

static const char *TAG = "ocr";

// TensorFlow Lite model and interpreter objects
static const tflite::Model* tflite_model = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;
static tflite::MicroErrorReporter error_reporter;
static TfLiteTensor* input_tensor = nullptr;
static TfLiteTensor* output_tensor = nullptr;

// Memory buffer for TensorFlow Lite (tensor arena)
// This needs to be allocated in 8-byte aligned memory
#define TENSOR_ARENA_SIZE (128 * 1024)
static uint8_t tensor_arena[TENSOR_ARENA_SIZE] __attribute__((aligned(8)));

// Current image buffer
static camera_fb_t* current_fb = NULL;
static uint8_t* processed_image = NULL;
static size_t processed_image_size = 0;

// Model parameters
static const int kNumClasses = 10; // Digits 0-9
static const int kImageWidth = 28;
static const int kImageHeight = 28;
static const int kImageChannels = 1; // Grayscale

// Forward declarations
static esp_err_t initialize_tflite_model(void);
static esp_err_t preprocess_image(void);
static esp_err_t run_inference(ocr_result_t* result);
static int get_highest_confidence_class(float* scores, int num_classes);
static void clear_processed_image(void);

/**
 * @brief Initialize the OCR engine
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t ocr_init(void)
{
    ESP_LOGI(TAG, "Initializing OCR engine");
    
    // Initialize the image processing component
    esp_err_t ret = image_processing_init();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize image processing");
        return ret;
    }
    
    // Initialize the TensorFlow Lite model
    ret = initialize_tflite_model();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize TensorFlow Lite model");
        return ret;
    }
    
    ESP_LOGI(TAG, "OCR engine initialized successfully");
    return ESP_OK;
}

/**
 * @brief Set the image for OCR processing
 * 
 * @param fb Camera frame buffer containing the image
 * @return esp_err_t ESP_OK on success
 */
esp_err_t ocr_set_image(camera_fb_t* fb)
{
    if (fb == NULL) {
        ESP_LOGE(TAG, "Invalid frame buffer");
        return ESP_ERR_INVALID_ARG;
    }
    
    // Store the frame buffer
    current_fb = fb;
    ESP_LOGI(TAG, "Image set for OCR processing: %dx%d, %d bytes", 
             fb->width, fb->height, fb->len);
    
    return ESP_OK;
}

/**
 * @brief Process the current image and extract meter reading
 * 
 * @param result Pointer to store the OCR result
 * @return esp_err_t ESP_OK on success
 */
esp_err_t ocr_process_image(ocr_result_t* result)
{
    ESP_LOGI(TAG, "Processing image for OCR");
    
    if (current_fb == NULL) {
        ESP_LOGE(TAG, "No image set for processing");
        return ESP_ERR_INVALID_STATE;
    }
    
    if (result == NULL) {
        ESP_LOGE(TAG, "Invalid result pointer");
        return ESP_ERR_INVALID_ARG;
    }
    
    // Start timing the OCR process
    int64_t start_time = esp_timer_get_time();
    
    // Preprocess the image
    esp_err_t ret = preprocess_image();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Image preprocessing failed");
        return ret;
    }
    
    // Run inference to extract digits
    ret = run_inference(result);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Inference failed");
        clear_processed_image();
        return ret;
    }
    
    // Calculate processing time
    int64_t end_time = esp_timer_get_time();
    float processing_time_ms = (end_time - start_time) / 1000.0f;
    
    ESP_LOGI(TAG, "OCR processing completed in %.2f ms", processing_time_ms);
    ESP_LOGI(TAG, "Recognized value: %s (Confidence: %d%%)", 
             result->text, result->confidence);
    
    // Clean up processed image
    clear_processed_image();
    
    return ESP_OK;
}

/**
 * @brief Initialize the TensorFlow Lite model
 * 
 * @return esp_err_t ESP_OK on success
 */
static esp_err_t initialize_tflite_model(void)
{
    ESP_LOGI(TAG, "Initializing TensorFlow Lite model");
    
    // Load the TensorFlow Lite model
    tflite_model = model_interface_get_model();
    if (tflite_model == nullptr) {
        ESP_LOGE(TAG, "Failed to load model");
        return ESP_FAIL;
    }
    
    // Set up the micro op resolver with the operations needed by the model
    static tflite::MicroMutableOpResolver<10> resolver;
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddFullyConnected();
    resolver.AddReshape();
    resolver.AddSoftmax();
    resolver.AddAdd();
    resolver.AddRelu();
    resolver.AddDequantize();
    resolver.AddQuantize();
    resolver.AddMul();
    
    // Create an interpreter to run the model
    static tflite::MicroInterpreter static_interpreter(
        tflite_model, resolver, tensor_arena, TENSOR_ARENA_SIZE, &error_reporter);
    interpreter = &static_interpreter;
    
    // Allocate memory for the model's tensors
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        ESP_LOGE(TAG, "Failed to allocate tensors: %d", allocate_status);
        return ESP_FAIL;
    }
    
    // Get pointers to the model's input and output tensors
    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);
    
    // Check tensor dimensions
    if (input_tensor->dims->size != 4 || 
        input_tensor->dims->data[1] != kImageHeight || 
        input_tensor->dims->data[2] != kImageWidth || 
        input_tensor->dims->data[3] != kImageChannels) {
        ESP_LOGE(TAG, "Unexpected input tensor dimensions. Expected %dx%dx%d, Got %dx%dx%d",
                 kImageHeight, kImageWidth, kImageChannels,
                 input_tensor->dims->data[1], input_tensor->dims->data[2], input_tensor->dims->data[3]);
        return ESP_FAIL;
    }
    
    ESP_LOGI(TAG, "TensorFlow Lite model initialized successfully");
    ESP_LOGI(TAG, "Input tensor size: %d x %d x %d", 
             input_tensor->dims->data[1], 
             input_tensor->dims->data[2], 
             input_tensor->dims->data[3]);
    ESP_LOGI(TAG, "Output tensor size: %d", output_tensor->dims->data[1]);
    
    return ESP_OK;
}

/**
 * @brief Preprocess the current image for OCR
 * 
 * @return esp_err_t ESP_OK on success
 */
static esp_err_t preprocess_image(void)
{
    ESP_LOGI(TAG, "Preprocessing image for OCR");
    
    if (current_fb == NULL) {
        ESP_LOGE(TAG, "No image set for processing");
        return ESP_ERR_INVALID_STATE;
    }
    
    // Free any previously processed image
    clear_processed_image();
    
    // First convert the image to grayscale, apply thresholding, etc.
    image_processing_result_t proc_result;
    esp_err_t ret = image_processing_prepare_for_ocr(current_fb, &proc_result);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Image preprocessing failed");
        return ret;
    }
    
    // Store the processed image
    processed_image = proc_result.processed_image;
    processed_image_size = proc_result.width * proc_result.height;
    
    // Log the ROI information
    ESP_LOGI(TAG, "Detected meter region: (%d,%d) -> (%d,%d)",
             proc_result.roi_x, proc_result.roi_y, 
             proc_result.roi_x + proc_result.roi_width,
             proc_result.roi_y + proc_result.roi_height);
    
    ESP_LOGI(TAG, "Image preprocessing completed");
    return ESP_OK;
}

/**
 * @brief Run inference on the preprocessed image
 * 
 * @param result Pointer to store the OCR result
 * @return esp_err_t ESP_OK on success
 */
static esp_err_t run_inference(ocr_result_t* result)
{
    ESP_LOGI(TAG, "Running inference");
    
    if (processed_image == NULL) {
        ESP_LOGE(TAG, "No preprocessed image available");
        return ESP_ERR_INVALID_STATE;
    }
    
    // Get the digit segmentation from the image processing step
    digit_segments_t segments;
    esp_err_t ret = image_processing_segment_digits(processed_image, 
                                                  processed_image_size, 
                                                  &segments);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Digit segmentation failed");
        return ret;
    }
    
    ESP_LOGI(TAG, "Found %d digit segments", segments.count);
    
    // Check if we have detected any digits
    if (segments.count == 0) {
        ESP_LOGW(TAG, "No digits detected in the image");
        result->text[0] = '\0';
        result->confidence = 0;
        return ESP_FAIL;
    }
    
    // Ensure we don't exceed the max digits
    if (segments.count > MAX_DIGITS) {
        ESP_LOGW(TAG, "Too many digits detected (%d), limiting to %d", 
                 segments.count, MAX_DIGITS);
        segments.count = MAX_DIGITS;
    }
    
    // Process each digit segment
    char digit_text[MAX_DIGITS + 1] = {0};
    int total_confidence = 0;
    
    for (int i = 0; i < segments.count; i++) {
        // Resize the digit to match the input tensor size
        uint8_t resized_digit[kImageHeight * kImageWidth];
        ret = image_processing_resize_digit(segments.segments[i].image,
                                          segments.segments[i].width,
                                          segments.segments[i].height,
                                          resized_digit,
                                          kImageWidth,
                                          kImageHeight);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "Failed to resize digit segment %d", i);
            continue;
        }
        
        // Copy the resized digit to the input tensor
        for (int y = 0; y < kImageHeight; y++) {
            for (int x = 0; x < kImageWidth; x++) {
                // Get the pixel value (0-255) and normalize to float (0.0-1.0)
                float pixel_value = resized_digit[y * kImageWidth + x] / 255.0f;
                input_tensor->data.f[y * kImageWidth + x] = pixel_value;
            }
        }
        
        // Run inference
        TfLiteStatus invoke_status = interpreter->Invoke();
        if (invoke_status != kTfLiteOk) {
            ESP_LOGE(TAG, "Invoke failed for digit %d", i);
            continue;
        }
        
        // Get the classification result (highest confidence class)
        int digit_class = get_highest_confidence_class(output_tensor->data.f, kNumClasses);
        int confidence = (int)(output_tensor->data.f[digit_class] * 100);
        
        // Add to the result
        digit_text[i] = '0' + digit_class;
        total_confidence += confidence;
        
        ESP_LOGI(TAG, "Digit %d classified as %d with confidence %d%%", 
                 i, digit_class, confidence);
    }
    
    // Calculate average confidence
    int avg_confidence = (segments.count > 0) ? total_confidence / segments.count : 0;
    
    // Sort the digits based on their X position (left to right reading)
    // This is simplified here; in a real implementation, we would sort the digits
    // based on their X position before classification
    
    // Copy the result
    strncpy(result->text, digit_text, MAX_DIGITS);
    result->text[segments.count] = '\0';
    result->confidence = avg_confidence;
    
    ESP_LOGI(TAG, "OCR result: %s (Confidence: %d%%)", result->text, result->confidence);
    return ESP_OK;
}

/**
 * @brief Get the class with highest confidence from inference output
 * 
 * @param scores Array of class scores
 * @param num_classes Number of classes
 * @return int Index of the class with highest confidence
 */
static int get_highest_confidence_class(float* scores, int num_classes)
{
    int highest_idx = 0;
    float highest_score = scores[0];
    
    for (int i = 1; i < num_classes; i++) {
        if (scores[i] > highest_score) {
            highest_score = scores[i];
            highest_idx = i;
        }
    }
    
    return highest_idx;
}

/**
 * @brief Clear the processed image buffer
 */
static void clear_processed_image(void)
{
    if (processed_image != NULL) {
        free(processed_image);
        processed_image = NULL;
        processed_image_size = 0;
    }
}

/**
 * @brief Clean up OCR resources
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t ocr_deinit(void)
{
    ESP_LOGI(TAG, "Deinitializing OCR engine");
    
    // Clean up processed image
    clear_processed_image();
    
    // Reset current frame buffer pointer
    current_fb = NULL;
    
    // Deinitialize image processing
    image_processing_deinit();
    
    ESP_LOGI(TAG, "OCR engine deinitialized");
    return ESP_OK;
}
