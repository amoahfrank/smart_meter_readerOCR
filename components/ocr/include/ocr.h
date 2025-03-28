/**
 * @file ocr.h
 * @brief OCR interface header for Smart Meter Reader OCR
 * 
 * This file defines the public interface for the OCR module,
 * which handles digit recognition for meter reading.
 * 
 * @author Smart Meter Reader OCR Team
 * @date 2023
 */

#ifndef OCR_H
#define OCR_H

#include "esp_err.h"
#include "esp_camera.h"

#define MAX_DIGITS 10   /*!< Maximum number of digits to recognize */

/**
 * @brief OCR result structure
 */
typedef struct {
    char text[MAX_DIGITS + 1];  /*!< Recognized text (null-terminated) */
    int confidence;             /*!< Recognition confidence (0-100) */
} ocr_result_t;

/**
 * @brief Initialize the OCR engine
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t ocr_init(void);

/**
 * @brief Set the image for OCR processing
 * 
 * @param fb Camera frame buffer containing the image
 * @return esp_err_t ESP_OK on success
 */
esp_err_t ocr_set_image(camera_fb_t* fb);

/**
 * @brief Process the current image and extract meter reading
 * 
 * @param result Pointer to store the OCR result
 * @return esp_err_t ESP_OK on success
 */
esp_err_t ocr_process_image(ocr_result_t* result);

/**
 * @brief Clean up OCR resources
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t ocr_deinit(void);

#endif /* OCR_H */
