/**
 * @file model_interface.c
 * @brief Implementation of the TensorFlow Lite model interface for OCR
 * 
 * This file implements the interface to load and interact with the TensorFlow 
 * Lite model for digit recognition.
 * 
 * @author Smart Meter Reader OCR Team
 * @date 2023
 */

#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_system.h"
#include "nvs_flash.h"
#include "esp_sleep.h"

// TensorFlow Lite includes
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Include local headers
#include "model_interface.h"

static const char *TAG = "model_interface";

// Model data is embedded in the firmware
extern const unsigned char digit_recognition_model_tflite_start[] asm("_binary_digit_recognition_model_tflite_start");
extern const unsigned char digit_recognition_model_tflite_end[] asm("_binary_digit_recognition_model_tflite_end");

// TensorFlow Lite model
static const tflite::Model* model = nullptr;

/**
 * @brief Initialize the model interface
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t model_interface_init(void)
{
    ESP_LOGI(TAG, "Initializing model interface");
    
    // Model is statically embedded in firmware
    model = tflite::GetModel(digit_recognition_model_tflite_start);
    
    // Verify model version
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema version %d is not equal to supported version %d",
                 model->version(), TFLITE_SCHEMA_VERSION);
        return ESP_FAIL;
    }
    
    ESP_LOGI(TAG, "Model interface initialized successfully");
    
    return ESP_OK;
}

/**
 * @brief Get the TensorFlow Lite model
 * 
 * @return const tflite::Model* Pointer to the model
 */
const tflite::Model* model_interface_get_model(void)
{
    if (model == nullptr) {
        ESP_LOGW(TAG, "Model interface not initialized, initializing now");
        esp_err_t ret = model_interface_init();
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "Failed to initialize model interface");
            return nullptr;
        }
    }
    
    return model;
}

/**
 * @brief Get model input shape
 * 
 * @param width Pointer to store input width
 * @param height Pointer to store input height
 * @param channels Pointer to store input channels
 * @return esp_err_t ESP_OK on success
 */
esp_err_t model_interface_get_input_shape(int *width, int *height, int *channels)
{
    if (model == nullptr) {
        ESP_LOGE(TAG, "Model interface not initialized");
        return ESP_ERR_INVALID_STATE;
    }
    
    if (width == NULL || height == NULL || channels == NULL) {
        ESP_LOGE(TAG, "Invalid parameters");
        return ESP_ERR_INVALID_ARG;
    }
    
    // For a pre-trained model, the input shape is known:
    // 28x28x1 (MNIST-like)
    *width = 28;
    *height = 28;
    *channels = 1;
    
    return ESP_OK;
}

/**
 * @brief Get the number of output classes
 * 
 * @param num_classes Pointer to store the number of classes
 * @return esp_err_t ESP_OK on success
 */
esp_err_t model_interface_get_num_classes(int *num_classes)
{
    if (model == nullptr) {
        ESP_LOGE(TAG, "Model interface not initialized");
        return ESP_ERR_INVALID_STATE;
    }
    
    if (num_classes == NULL) {
        ESP_LOGE(TAG, "Invalid parameter");
        return ESP_ERR_INVALID_ARG;
    }
    
    // For a pre-trained digit recognition model, the number of classes is 10 (digits 0-9)
    *num_classes = 10;
    
    return ESP_OK;
}

/**
 * @brief Get the required tensor arena size
 * 
 * @param arena_size Pointer to store the required arena size
 * @return esp_err_t ESP_OK on success
 */
esp_err_t model_interface_get_arena_size(size_t *arena_size)
{
    if (model == nullptr) {
        ESP_LOGE(TAG, "Model interface not initialized");
        return ESP_ERR_INVALID_STATE;
    }
    
    if (arena_size == NULL) {
        ESP_LOGE(TAG, "Invalid parameter");
        return ESP_ERR_INVALID_ARG;
    }
    
    // For the digit recognition model, the arena size is approximately 128KB
    // The exact size depends on the model architecture and quantization
    *arena_size = 128 * 1024;
    
    return ESP_OK;
}

/**
 * @brief Get the model version
 * 
 * @param version Buffer to store the version string
 * @param max_len Maximum length of the version buffer
 * @return esp_err_t ESP_OK on success
 */
esp_err_t model_interface_get_version(char *version, size_t max_len)
{
    if (model == nullptr) {
        ESP_LOGE(TAG, "Model interface not initialized");
        return ESP_ERR_INVALID_STATE;
    }
    
    if (version == NULL || max_len == 0) {
        ESP_LOGE(TAG, "Invalid parameters");
        return ESP_ERR_INVALID_ARG;
    }
    
    // For this implementation, use a fixed version
    const char *model_version = "1.0.0";
    
    if (strlen(model_version) >= max_len) {
        ESP_LOGE(TAG, "Version buffer too small");
        return ESP_ERR_INVALID_ARG;
    }
    
    strncpy(version, model_version, max_len - 1);
    version[max_len - 1] = '\0';
    
    return ESP_OK;
}

/**
 * @brief Deinitialize the model interface
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t model_interface_deinit(void)
{
    ESP_LOGI(TAG, "Deinitializing model interface");
    
    // Nothing to clean up, the model is statically embedded
    model = nullptr;
    
    ESP_LOGI(TAG, "Model interface deinitialized");
    
    return ESP_OK;
}
