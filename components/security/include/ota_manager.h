/**
 * @file ota_manager.h
 * @brief OTA update manager header for Smart Meter Reader OCR
 * 
 * This file defines the public interface for the OTA (Over-The-Air) update
 * manager, which handles firmware updates.
 * 
 * @author Smart Meter Reader OCR Team
 * @date 2023
 */

#ifndef OTA_MANAGER_H
#define OTA_MANAGER_H

#include "esp_err.h"

/**
 * @brief Initialize the OTA manager
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t ota_manager_init(void);

/**
 * @brief Set the OTA server URL
 * 
 * @param url The OTA server URL
 * @return esp_err_t ESP_OK on success
 */
esp_err_t ota_manager_set_server_url(const char *url);

/**
 * @brief Check if an OTA update is available
 * 
 * @return bool True if update is available
 */
bool ota_manager_check_update(void);

/**
 * @brief Perform OTA update
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t ota_manager_perform_update(void);

/**
 * @brief Deinitialize the OTA manager
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t ota_manager_deinit(void);

#endif /* OTA_MANAGER_H */
