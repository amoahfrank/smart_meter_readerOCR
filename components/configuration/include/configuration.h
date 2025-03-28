/**
 * @file configuration.h
 * @brief Configuration manager header for Smart Meter Reader OCR
 * 
 * This file defines the public interface for the configuration manager,
 * which handles device settings and configuration storage.
 * 
 * @author Smart Meter Reader OCR Team
 * @date 2023
 */

#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include "esp_err.h"

/**
 * @brief Communication mode enumeration
 */
typedef enum {
    COMM_MODE_WIFI = 0,  /*!< WiFi communication mode */
    COMM_MODE_BLE = 1,   /*!< BLE communication mode */
    COMM_MODE_LORA = 2   /*!< LoRaWAN communication mode */
} comm_mode_t;

/**
 * @brief Device configuration structure
 */
typedef struct {
    char wifi_ssid[32];           /*!< WiFi SSID */
    char wifi_password[64];        /*!< WiFi password */
    char server_url[128];          /*!< Server URL for data transmission */
    comm_mode_t comm_mode;         /*!< Communication mode */
    uint32_t reading_interval_sec; /*!< Reading interval in seconds */
    uint8_t ocr_min_confidence;    /*!< Minimum OCR confidence (0-100) */
} device_config_t;

/**
 * @brief Initialize the configuration manager
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t configuration_init(void);

/**
 * @brief Set default configuration values
 * 
 * @param config Pointer to configuration structure
 * @return esp_err_t ESP_OK on success
 */
esp_err_t configuration_set_defaults(device_config_t *config);

/**
 * @brief Load configuration from NVS
 * 
 * @param config Pointer to store configuration
 * @return esp_err_t ESP_OK on success
 */
esp_err_t configuration_load(device_config_t *config);

/**
 * @brief Save configuration to NVS
 * 
 * @param config Pointer to configuration to save
 * @return esp_err_t ESP_OK on success
 */
esp_err_t configuration_save(const device_config_t *config);

/**
 * @brief Deinitialize the configuration manager
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t configuration_deinit(void);

#endif /* CONFIGURATION_H */
