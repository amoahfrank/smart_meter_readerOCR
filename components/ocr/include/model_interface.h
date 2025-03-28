/**
 * @file model_interface.h
 * @brief TensorFlow Lite model interface header for Smart Meter Reader OCR
 * 
 * This file defines the public interface for interacting with the TensorFlow Lite
 * model used for digit recognition.
 * 
 * @author Smart Meter Reader OCR Team
 * @date 2023
 */

#ifndef MODEL_INTERFACE_H
#define MODEL_INTERFACE_H

#include "esp_err.h"
#include "tensorflow/lite/schema/schema_generated.h"

/**
 * @brief Initialize the model interface
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t model_interface_init(void);

/**
 * @brief Get the TensorFlow Lite model
 * 
 * @return const tflite::Model* Pointer to the model
 */
const tflite::Model* model_interface_get_model(void);

/**
 * @brief Get model input shape
 * 
 * @param width Pointer to store input width
 * @param height Pointer to store input height
 * @param channels Pointer to store input channels
 * @return esp_err_t ESP_OK on success
 */
esp_err_t model_interface_get_input_shape(int *width, int *height, int *channels);

/**
 * @brief Get the number of output classes
 * 
 * @param num_classes Pointer to store the number of classes
 * @return esp_err_t ESP_OK on success
 */
esp_err_t model_interface_get_num_classes(int *num_classes);

/**
 * @brief Get the required tensor arena size
 * 
 * @param arena_size Pointer to store the required arena size
 * @return esp_err_t ESP_OK on success
 */
esp_err_t model_interface_get_arena_size(size_t *arena_size);

/**
 * @brief Get the model version
 * 
 * @param version Buffer to store the version string
 * @param max_len Maximum length of the version buffer
 * @return esp_err_t ESP_OK on success
 */
esp_err_t model_interface_get_version(char *version, size_t max_len);

/**
 * @brief Deinitialize the model interface
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t model_interface_deinit(void);

#endif /* MODEL_INTERFACE_H */
