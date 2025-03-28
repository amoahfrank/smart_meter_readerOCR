/**
 * @file image_processing.h
 * @brief Image processing interface header for Smart Meter Reader OCR
 * 
 * This file defines the public interface for the image processing module,
 * which prepares images for OCR recognition.
 * 
 * @author Smart Meter Reader OCR Team
 * @date 2023
 */

#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include "esp_err.h"
#include "esp_camera.h"

#include "ocr.h"  // For MAX_DIGITS

/**
 * @brief Image processing result structure
 */
typedef struct {
    uint8_t *processed_image;  /*!< Pointer to processed image data */
    size_t width;              /*!< Width of the processed image */
    size_t height;             /*!< Height of the processed image */
    int roi_x;                 /*!< ROI x-coordinate in original image */
    int roi_y;                 /*!< ROI y-coordinate in original image */
    int roi_width;             /*!< ROI width in original image */
    int roi_height;            /*!< ROI height in original image */
} image_processing_result_t;

/**
 * @brief Digit segment structure
 */
typedef struct {
    uint8_t *image;     /*!< Pointer to segment image data */
    int x;              /*!< X-coordinate of segment in ROI */
    int y;              /*!< Y-coordinate of segment in ROI */
    int width;          /*!< Width of segment */
    int height;         /*!< Height of segment */
} digit_segment_t;

/**
 * @brief Digit segments container
 */
typedef struct {
    digit_segment_t segments[MAX_DIGITS];  /*!< Array of digit segments */
    int count;                             /*!< Number of segments */
} digit_segments_t;

/**
 * @brief Initialize the image processing module
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t image_processing_init(void);

/**
 * @brief Prepare an image for OCR processing
 * 
 * @param fb Input camera frame buffer
 * @param result Output structure with processed image and metadata
 * @return esp_err_t ESP_OK on success
 */
esp_err_t image_processing_prepare_for_ocr(const camera_fb_t *fb, image_processing_result_t *result);

/**
 * @brief Segment digits from a preprocessed image
 * 
 * @param image Preprocessed binary image
 * @param image_size Size of the image in bytes
 * @param segments Output structure with digit segments
 * @return esp_err_t ESP_OK on success
 */
esp_err_t image_processing_segment_digits(const uint8_t *image, size_t image_size, 
                                        digit_segments_t *segments);

/**
 * @brief Resize a digit image to the target dimensions
 * 
 * @param digit_image Input digit image
 * @param width Input width
 * @param height Input height
 * @param resized_image Output resized image (must be pre-allocated)
 * @param target_width Target width
 * @param target_height Target height
 * @return esp_err_t ESP_OK on success
 */
esp_err_t image_processing_resize_digit(const uint8_t *digit_image, size_t width, size_t height,
                                     uint8_t *resized_image, size_t target_width, size_t target_height);

/**
 * @brief Deinitialize the image processing module
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t image_processing_deinit(void);

#endif /* IMAGE_PROCESSING_H */
