/**
 * @file display.c
 * @brief Display module implementation for Smart Meter Reader OCR
 * 
 * This file implements the e-paper display interface for the Smart Meter Reader.
 * It handles display initialization, text rendering, status updates, and power
 * management for the e-paper display.
 * 
 * @author Smart Meter Reader OCR Team
 * @date 2023
 */

#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_timer.h"

// Include the GoodDisplay e-paper library
#include "epaper.h"
#include "epaper_fonts.h"

// Include local headers
#include "display.h"

static const char *TAG = "display";

// E-paper display dimensions
#define DISPLAY_WIDTH  200
#define DISPLAY_HEIGHT 200

// Define the pins for e-paper display
#define EPAPER_BUSY_PIN    GPIO_NUM_2
#define EPAPER_RESET_PIN   GPIO_NUM_4
#define EPAPER_DC_PIN      GPIO_NUM_5
#define EPAPER_CS_PIN      GPIO_NUM_15
#define EPAPER_CLK_PIN     GPIO_NUM_18
#define EPAPER_MOSI_PIN    GPIO_NUM_23

// Display buffer
static uint8_t *display_buffer = NULL;
static bool display_initialized = false;

// E-paper display configuration
static epaper_conf_t epaper_conf = {
    .reset_pin = EPAPER_RESET_PIN,
    .dc_pin = EPAPER_DC_PIN,
    .busy_pin = EPAPER_BUSY_PIN,
    .cs_pin = EPAPER_CS_PIN,
    .spi_host = HSPI_HOST,
    .spi_clk = EPAPER_CLK_PIN,
    .spi_mosi = EPAPER_MOSI_PIN,
    .width = DISPLAY_WIDTH,
    .height = DISPLAY_HEIGHT,
};

// Font definitions
static const Font_t *title_font = &Font16;
static const Font_t *normal_font = &Font12;
static const Font_t *small_font = &Font8;
static const Font_t *large_font = &Font24;

/**
 * @brief Initialize the display module
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t display_init(void)
{
    ESP_LOGI(TAG, "Initializing display");
    
    if (display_initialized) {
        ESP_LOGW(TAG, "Display already initialized");
        return ESP_OK;
    }
    
    // Allocate display buffer
    display_buffer = heap_caps_malloc(DISPLAY_WIDTH * DISPLAY_HEIGHT / 8, MALLOC_CAP_DMA);
    if (display_buffer == NULL) {
        ESP_LOGE(TAG, "Failed to allocate display buffer");
        return ESP_ERR_NO_MEM;
    }
    
    // Initialize e-paper display
    esp_err_t ret = epaper_init(&epaper_conf);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize e-paper display: %d", ret);
        free(display_buffer);
        display_buffer = NULL;
        return ret;
    }
    
    // Clear the display buffer
    memset(display_buffer, 0xFF, DISPLAY_WIDTH * DISPLAY_HEIGHT / 8);
    
    // Set display to initialized
    display_initialized = true;
    
    ESP_LOGI(TAG, "Display initialized successfully");
    return ESP_OK;
}

/**
 * @brief Clear the display buffer
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t display_clear(void)
{
    if (!display_initialized) {
        ESP_LOGE(TAG, "Display not initialized");
        return ESP_ERR_INVALID_STATE;
    }
    
    // Clear the display buffer (white)
    memset(display_buffer, 0xFF, DISPLAY_WIDTH * DISPLAY_HEIGHT / 8);
    
    return ESP_OK;
}

/**
 * @brief Update the display with current buffer contents
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t display_update(void)
{
    if (!display_initialized) {
        ESP_LOGE(TAG, "Display not initialized");
        return ESP_ERR_INVALID_STATE;
    }
    
    ESP_LOGI(TAG, "Updating display");
    
    // Send the buffer to the display
    esp_err_t ret = epaper_display_full(display_buffer);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to update display: %d", ret);
        return ret;
    }
    
    return ESP_OK;
}

/**
 * @brief Update only part of the display (faster)
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t display_partial_update(void)
{
    if (!display_initialized) {
        ESP_LOGE(TAG, "Display not initialized");
        return ESP_ERR_INVALID_STATE;
    }
    
    ESP_LOGI(TAG, "Partial display update");
    
    // Send the buffer to the display using partial update
    esp_err_t ret = epaper_display_partial(display_buffer);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to partially update display: %d", ret);
        return ret;
    }
    
    return ESP_OK;
}

/**
 * @brief Show splash screen
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t display_show_splash(void)
{
    if (!display_initialized) {
        ESP_LOGE(TAG, "Display not initialized");
        return ESP_ERR_INVALID_STATE;
    }
    
    // Clear the display
    display_clear();
    
    // Draw the splash screen
    epaper_draw_string(display_buffer, 20, 40, "Smart Meter", large_font, BLACK);
    epaper_draw_string(display_buffer, 30, 80, "Reader OCR", large_font, BLACK);
    epaper_draw_string(display_buffer, 25, 140, "Initializing...", normal_font, BLACK);
    
    // Draw a border
    epaper_draw_rect(display_buffer, 5, 5, DISPLAY_WIDTH - 10, DISPLAY_HEIGHT - 10, BLACK);
    
    // Update the display
    return display_update();
}

/**
 * @brief Show text on the display
 * 
 * @param line1 First line of text (title)
 * @param line2 Second line of text (main)
 * @param line3 Third line of text (optional, can be NULL)
 * @param ... Format arguments for line3 if it's a format string
 * @return esp_err_t ESP_OK on success
 */
esp_err_t display_show_text(const char *line1, const char *line2, const char *line3, ...)
{
    if (!display_initialized) {
        ESP_LOGE(TAG, "Display not initialized");
        return ESP_ERR_INVALID_STATE;
    }
    
    // Draw the first line (title)
    if (line1 != NULL) {
        epaper_draw_string(display_buffer, 10, 20, line1, title_font, BLACK);
    }
    
    // Draw the second line (main text)
    if (line2 != NULL) {
        epaper_draw_string(display_buffer, 10, 60, line2, normal_font, BLACK);
    }
    
    // Draw the third line (optional)
    if (line3 != NULL) {
        char buffer[128];
        
        va_list args;
        va_start(args, line3);
        vsnprintf(buffer, sizeof(buffer), line3, args);
        va_end(args);
        
        epaper_draw_string(display_buffer, 10, 100, buffer, normal_font, BLACK);
    }
    
    return ESP_OK;
}

/**
 * @brief Show meter reading on the display
 * 
 * @param reading Meter reading text
 * @param timestamp Timestamp string
 * @param status Status string
 * @return esp_err_t ESP_OK on success
 */
esp_err_t display_show_reading(const char *reading, const char *timestamp, const char *status)
{
    if (!display_initialized) {
        ESP_LOGE(TAG, "Display not initialized");
        return ESP_ERR_INVALID_STATE;
    }
    
    // Clear the display
    display_clear();
    
    // Draw a border
    epaper_draw_rect(display_buffer, 5, 5, DISPLAY_WIDTH - 10, DISPLAY_HEIGHT - 10, BLACK);
    
    // Draw the title
    epaper_draw_string(display_buffer, 20, 20, "Meter Reading", title_font, BLACK);
    
    // Draw a separator line
    epaper_draw_line(display_buffer, 10, 40, DISPLAY_WIDTH - 10, 40, BLACK);
    
    // Draw the reading in large font
    if (reading != NULL) {
        // Center the reading
        int width = epaper_get_string_width(reading, large_font);
        int x_pos = (DISPLAY_WIDTH - width) / 2;
        epaper_draw_string(display_buffer, x_pos, 80, reading, large_font, BLACK);
    }
    
    // Draw timestamp
    if (timestamp != NULL) {
        epaper_draw_string(display_buffer, 10, 140, timestamp, small_font, BLACK);
    }
    
    // Draw status (e.g., battery level)
    if (status != NULL) {
        int width = epaper_get_string_width(status, small_font);
        epaper_draw_string(display_buffer, DISPLAY_WIDTH - width - 10, 140, status, small_font, BLACK);
    }
    
    return ESP_OK;
}

/**
 * @brief Show status message at the bottom of the display
 * 
 * @param status Status message
 * @return esp_err_t ESP_OK on success
 */
esp_err_t display_show_status(const char *status)
{
    if (!display_initialized) {
        ESP_LOGE(TAG, "Display not initialized");
        return ESP_ERR_INVALID_STATE;
    }
    
    if (status == NULL) {
        return ESP_OK;
    }
    
    // Fill the status area with white first
    epaper_fill_rect(display_buffer, 10, 160, DISPLAY_WIDTH - 20, 30, WHITE);
    
    // Draw the status message
    epaper_draw_string(display_buffer, 10, 170, status, normal_font, BLACK);
    
    return ESP_OK;
}

/**
 * @brief Deinitialize the display to save power
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t display_deinit(void)
{
    ESP_LOGI(TAG, "Deinitializing display");
    
    if (!display_initialized) {
        ESP_LOGW(TAG, "Display not initialized");
        return ESP_OK;
    }
    
    // Put the display in sleep mode
    esp_err_t ret = epaper_sleep();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to put display to sleep: %d", ret);
        return ret;
    }
    
    // Free the display buffer
    if (display_buffer != NULL) {
        free(display_buffer);
        display_buffer = NULL;
    }
    
    display_initialized = false;
    
    ESP_LOGI(TAG, "Display deinitialized successfully");
    return ESP_OK;
}
