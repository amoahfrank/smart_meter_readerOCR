/**
 * @file camera.c
 * @brief Camera module implementation for Smart Meter Reader OCR
 * 
 * This file implements the camera interface for capturing images of meter
 * displays. It handles camera initialization, configuration, image capture,
 * and basic image processing functions.
 * 
 * @author Smart Meter Reader OCR Team
 * @date 2023
 */

#include <stdio.h>
#include <string.h>
#include "esp_log.h"
#include "esp_camera.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

// Include local headers
#include "camera.h"

static const char *TAG = "camera";

// Camera configuration
static camera_config_t camera_config;

// Current camera state
static bool camera_initialized = false;
static int retry_count = 0;

// Camera pins for ESP32-S3 (adjust as needed for your hardware)
#define CAM_PIN_PWDN    -1
#define CAM_PIN_RESET   -1
#define CAM_PIN_XCLK    45
#define CAM_PIN_SIOD    1
#define CAM_PIN_SIOC    2
#define CAM_PIN_D7      12
#define CAM_PIN_D6      11
#define CAM_PIN_D5      10
#define CAM_PIN_D4      9
#define CAM_PIN_D3      8
#define CAM_PIN_D2      7
#define CAM_PIN_D1      6
#define CAM_PIN_D0      5
#define CAM_PIN_VSYNC   4
#define CAM_PIN_HREF    3
#define CAM_PIN_PCLK    13

/**
 * @brief Initialize the camera with the provided configuration
 * 
 * @param config Camera initialization configuration
 * @return esp_err_t ESP_OK on success
 */
esp_err_t camera_init(const camera_init_config_t *config)
{
    ESP_LOGI(TAG, "Initializing camera");
    
    if (camera_initialized) {
        ESP_LOGW(TAG, "Camera already initialized");
        return ESP_OK;
    }
    
    // Set up camera configuration
    camera_config.pin_pwdn = CAM_PIN_PWDN;
    camera_config.pin_reset = CAM_PIN_RESET;
    camera_config.pin_xclk = CAM_PIN_XCLK;
    camera_config.pin_sccb_sda = CAM_PIN_SIOD;
    camera_config.pin_sccb_scl = CAM_PIN_SIOC;
    camera_config.pin_d7 = CAM_PIN_D7;
    camera_config.pin_d6 = CAM_PIN_D6;
    camera_config.pin_d5 = CAM_PIN_D5;
    camera_config.pin_d4 = CAM_PIN_D4;
    camera_config.pin_d3 = CAM_PIN_D3;
    camera_config.pin_d2 = CAM_PIN_D2;
    camera_config.pin_d1 = CAM_PIN_D1;
    camera_config.pin_d0 = CAM_PIN_D0;
    camera_config.pin_vsync = CAM_PIN_VSYNC;
    camera_config.pin_href = CAM_PIN_HREF;
    camera_config.pin_pclk = CAM_PIN_PCLK;
    
    // Set clock frequency
    camera_config.xclk_freq_hz = config->xclk_freq_hz;
    
    // Set pixel format
    camera_config.pixel_format = config->pixel_format;
    
    // Set frame size
    camera_config.frame_size = config->frame_size;
    
    // Set JPEG quality
    camera_config.jpeg_quality = config->jpeg_quality;
    
    // Set number of frame buffers
    camera_config.fb_count = config->fb_count;
    
    // XCLK signal needs to be provided to camera
    ESP_LOGI(TAG, "Configuring camera clock");
    
    // Initialize the camera
    esp_err_t err = esp_camera_init(&camera_config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Camera initialization failed with error 0x%x", err);
        return err;
    }
    
    // Set initial camera parameters
    sensor_t *sensor = esp_camera_sensor_get();
    if (sensor) {
        // Adjust for meter reading - we want high contrast, clear images
        sensor->set_brightness(sensor, 0);     // -2 to 2
        sensor->set_contrast(sensor, 1);       // -2 to 2
        sensor->set_saturation(sensor, 0);     // -2 to 2
        sensor->set_sharpness(sensor, 1);      // -2 to 2
        sensor->set_gain_ctrl(sensor, 1);      // Auto gain on
        sensor->set_exposure_ctrl(sensor, 1);  // Auto exposure on
        sensor->set_whitebal(sensor, 1);       // Auto white balance on
        sensor->set_awb_gain(sensor, 1);       // Auto white balance gain on
        sensor->set_aec2(sensor, 1);           // Auto exposure correction on
        sensor->set_ae_level(sensor, 0);       // -2 to 2
        sensor->set_aec_value(sensor, 300);    // 0 to 1200
        sensor->set_denoise(sensor, 1);        // Denoise on
        
        // Specific settings for meter reading
        sensor->set_quality(sensor, 10);       // 10-63, lower means higher quality
        sensor->set_colorbar(sensor, 0);       // 0 = disable, 1 = enable
        sensor->set_special_effect(sensor, 0); // 0 = no effect
        sensor->set_hmirror(sensor, 0);        // 0 = disable, 1 = enable
        sensor->set_vflip(sensor, 0);          // 0 = disable, 1 = enable
        
        ESP_LOGI(TAG, "Camera sensor configured for meter reading");
    }
    
    // Mark as initialized
    camera_initialized = true;
    retry_count = 0;
    
    ESP_LOGI(TAG, "Camera initialized successfully");
    return ESP_OK;
}

/**
 * @brief Deinitialize the camera to save power
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t camera_deinit(void)
{
    ESP_LOGI(TAG, "Deinitializing camera");
    
    if (!camera_initialized) {
        ESP_LOGW(TAG, "Camera not initialized");
        return ESP_OK;
    }
    
    // Deinitialize the camera
    esp_err_t err = esp_camera_deinit();
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Camera deinitialization failed with error 0x%x", err);
        return err;
    }
    
    camera_initialized = false;
    ESP_LOGI(TAG, "Camera deinitialized successfully");
    return ESP_OK;
}

/**
 * @brief Capture a frame from the camera
 * 
 * @return camera_fb_t* Pointer to the frame buffer, or NULL on failure
 */
camera_fb_t* camera_capture(void)
{
    ESP_LOGI(TAG, "Capturing image");
    
    if (!camera_initialized) {
        ESP_LOGE(TAG, "Camera not initialized");
        return NULL;
    }
    
    // Capture a frame
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        ESP_LOGE(TAG, "Failed to capture image");
        return NULL;
    }
    
    ESP_LOGI(TAG, "Image captured: %dx%d, %d bytes", 
             fb->width, fb->height, fb->len);
    
    return fb;
}

/**
 * @brief Return a frame buffer to the pool
 * 
 * @param fb Pointer to the frame buffer
 */
void camera_return_fb(camera_fb_t *fb)
{
    if (fb) {
        esp_camera_fb_return(fb);
    }
}

/**
 * @brief Adjust camera settings for retry after low OCR confidence
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t camera_adjust_for_retry(void)
{
    ESP_LOGI(TAG, "Adjusting camera for retry (count: %d)", retry_count);
    
    if (!camera_initialized) {
        ESP_LOGE(TAG, "Camera not initialized");
        return ESP_ERR_INVALID_STATE;
    }
    
    sensor_t *sensor = esp_camera_sensor_get();
    if (!sensor) {
        ESP_LOGE(TAG, "Failed to get camera sensor");
        return ESP_FAIL;
    }
    
    // Adjust settings based on retry count
    retry_count++;
    
    switch (retry_count) {
        case 1:
            // First retry: Increase contrast and brightness
            sensor->set_contrast(sensor, 2);      // Max contrast
            sensor->set_brightness(sensor, 1);    // Slightly brighter
            sensor->set_aec_value(sensor, 400);   // Increase exposure
            break;
            
        case 2:
            // Second retry: Decrease brightness, try different exposure
            sensor->set_brightness(sensor, -1);   // Slightly darker
            sensor->set_aec_value(sensor, 200);   // Decrease exposure
            sensor->set_gain_ctrl(sensor, 0);     // Manual gain
            sensor->set_agc_gain(sensor, 2);      // Set gain manually
            break;
            
        case 3:
            // Third retry: Try with flash if available, or max settings
            sensor->set_brightness(sensor, 0);    // Normal brightness
            sensor->set_contrast(sensor, 2);      // Max contrast
            sensor->set_saturation(sensor, -2);   // Minimum saturation
            sensor->set_aec_value(sensor, 300);   // Normal exposure
            
            // If we have LED flash control, use it
            #ifdef CONFIG_LED_ILLUMINATOR_ENABLED
            sensor->set_led_intensity(sensor, 255);  // Full LED intensity
            #endif
            break;
            
        default:
            // Reset to default settings after multiple tries
            sensor->set_brightness(sensor, 0);
            sensor->set_contrast(sensor, 1);
            sensor->set_saturation(sensor, 0);
            sensor->set_gain_ctrl(sensor, 1);
            sensor->set_exposure_ctrl(sensor, 1);
            sensor->set_aec_value(sensor, 300);
            retry_count = 0;
            
            #ifdef CONFIG_LED_ILLUMINATOR_ENABLED
            sensor->set_led_intensity(sensor, 0);   // Turn off LED
            #endif
            break;
    }
    
    // Wait for settings to take effect
    vTaskDelay(pdMS_TO_TICKS(100));
    
    ESP_LOGI(TAG, "Camera adjusted for retry");
    return ESP_OK;
}

/**
 * @brief Get current frame size
 * 
 * @return camera_framesize_t Current frame size
 */
camera_framesize_t camera_get_frame_size(void)
{
    return camera_config.frame_size;
}

/**
 * @brief Set frame size
 * 
 * @param frame_size New frame size
 * @return esp_err_t ESP_OK on success
 */
esp_err_t camera_set_frame_size(camera_framesize_t frame_size)
{
    if (!camera_initialized) {
        ESP_LOGE(TAG, "Camera not initialized");
        return ESP_ERR_INVALID_STATE;
    }
    
    sensor_t *sensor = esp_camera_sensor_get();
    if (!sensor) {
        ESP_LOGE(TAG, "Failed to get camera sensor");
        return ESP_FAIL;
    }
    
    ESP_LOGI(TAG, "Setting frame size to %d", frame_size);
    return sensor->set_framesize(sensor, frame_size);
}

/**
 * @brief Test camera functionality
 * 
 * @return esp_err_t ESP_OK if camera is working properly
 */
esp_err_t camera_test(void)
{
    if (!camera_initialized) {
        ESP_LOGE(TAG, "Camera not initialized");
        return ESP_ERR_INVALID_STATE;
    }
    
    // Try to capture a test frame
    camera_fb_t *fb = camera_capture();
    if (!fb) {
        ESP_LOGE(TAG, "Camera test failed - could not capture image");
        return ESP_FAIL;
    }
    
    // Check frame properties
    if (fb->width == 0 || fb->height == 0 || fb->len == 0) {
        ESP_LOGE(TAG, "Camera test failed - invalid frame properties");
        camera_return_fb(fb);
        return ESP_FAIL;
    }
    
    // Return the frame buffer
    camera_return_fb(fb);
    
    ESP_LOGI(TAG, "Camera test passed");
    return ESP_OK;
}
