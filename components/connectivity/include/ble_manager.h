/**
 * @file ble_manager.h
 * @brief BLE connectivity manager header
 * 
 * This file defines the public interface for the BLE connectivity manager,
 * which handles BLE connection, configuration, and data transmission.
 * 
 * @author Frank Amoah A.K.A SiRWaTT Smart Meter Reader OCR Team Lead
 * @date 2023
 */

#ifndef BLE_MANAGER_H
#define BLE_MANAGER_H

#include "esp_err.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"

#include "app_main.h"  // For system_state_t

/**
 * @brief Initialize the BLE manager
 * 
 * @param evt_group Event group for signaling
 * @return esp_err_t ESP_OK on success
 */
esp_err_t ble_manager_init(EventGroupHandle_t evt_group);

/**
 * @brief Start configuration mode via BLE
 * 
 * @param state System state structure
 * @return esp_err_t ESP_OK on success
 */
esp_err_t ble_manager_start_config_mode(system_state_t *state);

/**
 * @brief Stop configuration mode
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t ble_manager_stop_config_mode(void);

/**
 * @brief Transmit data via BLE
 * 
 * @param data Data to transmit (JSON string)
 * @return esp_err_t ESP_OK on success
 */
esp_err_t ble_manager_transmit_data(const char *data);

/**
 * @brief Deinitialize BLE manager
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t ble_manager_deinit(void);

#endif /* BLE_MANAGER_H */
