/**
 * @file wifi_manager.h
 * @brief WiFi connectivity manager header
 * 
 * This file defines the public interface for the WiFi connectivity manager,
 * which handles WiFi connection and data transmission.
 * 
 * @author Smart Meter Reader OCR Team
 * @date 2023
 */

#ifndef WIFI_MANAGER_H
#define WIFI_MANAGER_H

#include "esp_err.h"

/**
 * @brief Initialize the WiFi subsystem
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t wifi_manager_init(void);

/**
 * @brief Connect to WiFi network
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t wifi_manager_connect(void);

/**
 * @brief Disconnect from WiFi network
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t wifi_manager_disconnect(void);

/**
 * @brief Set WiFi credentials
 * 
 * @param ssid WiFi SSID
 * @param password WiFi password
 * @return esp_err_t ESP_OK on success
 */
esp_err_t wifi_manager_set_credentials(const char *ssid, const char *password);

/**
 * @brief Set server URL
 * 
 * @param url Server URL
 * @return esp_err_t ESP_OK on success
 */
esp_err_t wifi_manager_set_server_url(const char *url);

/**
 * @brief Transmit data to server
 * 
 * @param data Data to transmit (JSON string)
 * @return esp_err_t ESP_OK on success
 */
esp_err_t wifi_manager_transmit_data(const char *data);

/**
 * @brief Deinitialize WiFi manager
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t wifi_manager_deinit(void);

#endif /* WIFI_MANAGER_H */
