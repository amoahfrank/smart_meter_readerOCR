/**
 * @file lora_manager.h
 * @brief LoRaWAN connectivity manager header
 * 
 * This file defines the public interface for the LoRaWAN connectivity manager,
 * which handles LoRa connection and data transmission.
 * 
 * @author Smart Meter Reader OCR Team
 * @date 2023
 */

#ifndef LORA_MANAGER_H
#define LORA_MANAGER_H

#include "esp_err.h"

/**
 * @brief Initialize the LoRaWAN manager
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t lora_manager_init(void);

/**
 * @brief Set the LoRaWAN keys
 * 
 * @param dev_eui_hex DevEUI in hex format
 * @param app_eui_hex AppEUI in hex format
 * @param app_key_hex AppKey in hex format
 * @return esp_err_t ESP_OK on success
 */
esp_err_t lora_manager_set_keys(const char *dev_eui_hex, const char *app_eui_hex, const char *app_key_hex);

/**
 * @brief Transmit data over LoRaWAN
 * 
 * @param data Data to transmit
 * @return esp_err_t ESP_OK on success
 */
esp_err_t lora_manager_transmit_data(const char *data);

/**
 * @brief Deinitialize the LoRaWAN manager
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t lora_manager_deinit(void);

#endif /* LORA_MANAGER_H */
