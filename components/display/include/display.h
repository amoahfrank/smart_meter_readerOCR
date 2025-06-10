/**
 * @file display.h
 * @brief Display module header for Smart Meter Reader OCR
 * 
 * This file defines the public interface for the display module,
 * which manages the e-paper display functionality.
 * 
 * @author Frank Amoah A.K.A SiRWaTT Smart Meter Reader OCR Team Lead
 * @date 2023
 */

#ifndef DISPLAY_H
#define DISPLAY_H

#include "esp_err.h"

/**
 * @brief Initialize the display module
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t display_init(void);

/**
 * @brief Clear the display buffer
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t display_clear(void);

/**
 * @brief Update the display with current buffer contents
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t display_update(void);

/**
 * @brief Update only part of the display (faster)
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t display_partial_update(void);

/**
 * @brief Show splash screen
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t display_show_splash(void);

/**
 * @brief Show text on the display
 * 
 * @param line1 First line of text (title)
 * @param line2 Second line of text (main)
 * @param line3 Third line of text (optional, can be NULL)
 * @param ... Format arguments for line3 if it's a format string
 * @return esp_err_t ESP_OK on success
 */
esp_err_t display_show_text(const char *line1, const char *line2, const char *line3, ...);

/**
 * @brief Show meter reading on the display
 * 
 * @param reading Meter reading text
 * @param timestamp Timestamp string
 * @param status Status string
 * @return esp_err_t ESP_OK on success
 */
esp_err_t display_show_reading(const char *reading, const char *timestamp, const char *status);

/**
 * @brief Show status message at the bottom of the display
 * 
 * @param status Status message
 * @return esp_err_t ESP_OK on success
 */
esp_err_t display_show_status(const char *status);

/**
 * @brief Deinitialize the display to save power
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t display_deinit(void);

#endif /* DISPLAY_H */
