/**
 * @file ocr.h
 * @brief Main OCR interface for smart meter reading
 * 
 * This file provides the main interface for the OCR system, including
 * initialization, image processing, digit recognition, and result management.
 * 
 * @author Smart Meter Reader OCR Team
 * @date 2023
 */

#ifndef OCR_H
#define OCR_H

#include <stdint.h>
#include <stdbool.h>
#include "esp_err.h"
#include "esp_camera.h"
#include "image_processing.h"
#include "model_interface.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Maximum number of digits that can be recognized in a single reading
 */
#define OCR_MAX_DIGITS 10

/**
 * @brief Maximum length of a meter reading string (including null terminator)
 */
#define OCR_MAX_READING_LENGTH (OCR_MAX_DIGITS + 1)

/**
 * @brief OCR confidence threshold (0.0 to 1.0)
 */
#define OCR_MIN_CONFIDENCE 0.7f

/**
 * @brief OCR result structure
 */
typedef struct {
    char reading[OCR_MAX_READING_LENGTH];   ///< Recognized reading as string
    float confidence;                        ///< Overall confidence score (0.0 to 1.0)
    int digit_count;                        ///< Number of digits recognized
    float digit_confidences[OCR_MAX_DIGITS]; ///< Individual digit confidence scores
    uint32_t processing_time_ms;            ///< Total processing time in milliseconds
    bool is_valid;                          ///< Whether the reading is considered valid
} ocr_result_t;

/**
 * @brief OCR configuration structure
 */
typedef struct {
    float confidence_threshold;             ///< Minimum confidence threshold
    bool enable_preprocessing;              ///< Enable image preprocessing
    bool enable_digit_filtering;            ///< Enable digit validation/filtering
    int max_digits;                        ///< Maximum number of digits to recognize
    bool debug_mode;                       ///< Enable debug output and logging
} ocr_config_t;

/**
 * @brief OCR statistics structure
 */
typedef struct {
    uint32_t total_readings;               ///< Total number of readings attempted
    uint32_t successful_readings;          ///< Number of successful readings
    uint32_t failed_readings;              ///< Number of failed readings
    float average_confidence;              ///< Average confidence score
    uint32_t average_processing_time_ms;   ///< Average processing time
    uint32_t uptime_seconds;              ///< System uptime in seconds
} ocr_stats_t;

/**
 * @brief Initialize the OCR system
 * 
 * This function initializes all OCR components including image processing,
 * TensorFlow Lite model, and internal data structures.
 * 
 * @param config OCR configuration parameters
 * @return esp_err_t ESP_OK on success, error code otherwise
 */
esp_err_t ocr_init(const ocr_config_t *config);

/**
 * @brief Process a camera frame and extract meter reading
 * 
 * This is the main OCR function that takes a camera frame and returns
 * the recognized meter reading with confidence scores.
 * 
 * @param fb Camera frame buffer
 * @param result Output structure containing the OCR result
 * @return esp_err_t ESP_OK on success, error code otherwise
 */
esp_err_t ocr_process_frame(const camera_fb_t *fb, ocr_result_t *result);

/**
 * @brief Process a raw image buffer and extract meter reading
 * 
 * Alternative to ocr_process_frame() for processing pre-loaded images.
 * 
 * @param image_data Raw image data
 * @param width Image width
 * @param height Image height
 * @param format Image format (RGB, grayscale, etc.)
 * @param result Output structure containing the OCR result
 * @return esp_err_t ESP_OK on success, error code otherwise
 */
esp_err_t ocr_process_image(const uint8_t *image_data, size_t width, size_t height, 
                           pixformat_t format, ocr_result_t *result);

/**
 * @brief Update OCR configuration
 * 
 * @param config New configuration parameters
 * @return esp_err_t ESP_OK on success, error code otherwise
 */
esp_err_t ocr_update_config(const ocr_config_t *config);

/**
 * @brief Get current OCR configuration
 * 
 * @param config Output buffer for current configuration
 * @return esp_err_t ESP_OK on success, error code otherwise
 */
esp_err_t ocr_get_config(ocr_config_t *config);

/**
 * @brief Get OCR system statistics
 * 
 * @param stats Output buffer for statistics
 * @return esp_err_t ESP_OK on success, error code otherwise
 */
esp_err_t ocr_get_stats(ocr_stats_t *stats);

/**
 * @brief Reset OCR statistics
 * 
 * @return esp_err_t ESP_OK on success, error code otherwise
 */
esp_err_t ocr_reset_stats(void);

/**
 * @brief Validate a reading string
 * 
 * Performs basic validation on a reading string to check if it's reasonable
 * for a meter reading (e.g., not all zeros, reasonable length, etc.).
 * 
 * @param reading Reading string to validate
 * @param confidence Overall confidence score
 * @return bool true if reading is valid, false otherwise
 */
bool ocr_validate_reading(const char *reading, float confidence);

/**
 * @brief Enable or disable debug mode
 * 
 * @param enable true to enable debug mode, false to disable
 * @return esp_err_t ESP_OK on success, error code otherwise
 */
esp_err_t ocr_set_debug_mode(bool enable);

/**
 * @brief Save debug images to filesystem (if available)
 * 
 * When debug mode is enabled, this function can save intermediate processing
 * images to help with troubleshooting and model improvement.
 * 
 * @param prefix Filename prefix for saved images
 * @return esp_err_t ESP_OK on success, error code otherwise
 */
esp_err_t ocr_save_debug_images(const char *prefix);

/**
 * @brief Perform OCR system self-test
 * 
 * Runs internal tests to verify that the OCR system is functioning correctly.
 * 
 * @return esp_err_t ESP_OK if all tests pass, error code otherwise
 */
esp_err_t ocr_self_test(void);

/**
 * @brief Cleanup and deinitialize the OCR system
 * 
 * @return esp_err_t ESP_OK on success, error code otherwise
 */
esp_err_t ocr_deinit(void);

/**
 * @brief Get default OCR configuration
 * 
 * @return ocr_config_t Default configuration structure
 */
ocr_config_t ocr_get_default_config(void);

/**
 * @brief Convert OCR result to JSON string
 * 
 * @param result OCR result to convert
 * @param json_buffer Output buffer for JSON string
 * @param buffer_size Size of the output buffer
 * @return esp_err_t ESP_OK on success, error code otherwise
 */
esp_err_t ocr_result_to_json(const ocr_result_t *result, char *json_buffer, size_t buffer_size);

#ifdef __cplusplus
}
#endif

#endif // OCR_H
