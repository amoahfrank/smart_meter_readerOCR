/**
 * @file camera.h
 * @brief Camera module header for Smart Meter Reader OCR
 * 
 * This file defines the public interface for the camera module,
 * which handles image capture for meter reading.
 * 
 * @author Frank Amoah A.K.A SiRWaTT Smart Meter Reader OCR Team Lead
 * @date 2023
 */

#ifndef CAMERA_H
#define CAMERA_H

#include "esp_err.h"
#include "esp_camera.h"

/**
 * @brief Camera initialization configuration
 */
typedef struct {
    int xclk_freq_hz;                /*!< XCLK frequency in Hz */
    pixformat_t pixel_format;        /*!< Pixel format */
    framesize_t frame_size;          /*!< Frame size */
    int jpeg_quality;                /*!< JPEG quality (0-63, lower value means higher quality) */
    size_t fb_count;                 /*!< Number of frame buffers */
} camera_init_config_t;

// Default camera configuration
#define CAMERA_PIXEL_FORMAT     PIXFORMAT_RGB565
#define CAMERA_FRAME_SIZE       FRAMESIZE_VGA
#define CONFIG_XCLK_FREQ        20000000   // 20MHz XCLK frequency

/**
 * @brief Initialize the camera with the provided configuration
 * 
 * @param config Camera initialization configuration
 * @return esp_err_t ESP_OK on success
 */
esp_err_t camera_init(const camera_init_config_t *config);

/**
 * @brief Deinitialize the camera to save power
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t camera_deinit(void);

/**
 * @brief Capture a frame from the camera
 * 
 * @return camera_fb_t* Pointer to the frame buffer, or NULL on failure
 */
camera_fb_t* camera_capture(void);

/**
 * @brief Return a frame buffer to the pool
 * 
 * @param fb Pointer to the frame buffer
 */
void camera_return_fb(camera_fb_t *fb);

/**
 * @brief Adjust camera settings for retry after low OCR confidence
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t camera_adjust_for_retry(void);

/**
 * @brief Get current frame size
 * 
 * @return camera_framesize_t Current frame size
 */
camera_framesize_t camera_get_frame_size(void);

/**
 * @brief Set frame size
 * 
 * @param frame_size New frame size
 * @return esp_err_t ESP_OK on success
 */
esp_err_t camera_set_frame_size(camera_framesize_t frame_size);

/**
 * @brief Test camera functionality
 * 
 * @return esp_err_t ESP_OK if camera is working properly
 */
esp_err_t camera_test(void);

#endif /* CAMERA_H */
