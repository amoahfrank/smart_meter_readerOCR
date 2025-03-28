/**
 * @file power_mgmt.h
 * @brief Power management header for Smart Meter Reader OCR
 * 
 * This file defines the public interface for the power management module,
 * which handles battery monitoring and power saving features.
 * 
 * @author Smart Meter Reader OCR Team
 * @date 2023
 */

#ifndef POWER_MGMT_H
#define POWER_MGMT_H

#include "esp_err.h"

// Critical battery threshold in percentage
#define BATTERY_CRITICAL_THRESHOLD  10

/**
 * @brief Initialize the power management module
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t power_mgmt_init(void);

/**
 * @brief Get the current battery level as a percentage
 * 
 * @return int Battery level (0-100)
 */
int power_mgmt_get_battery_level(void);

/**
 * @brief Get the current battery voltage in millivolts
 * 
 * @return int Battery voltage (mV)
 */
int power_mgmt_get_battery_voltage(void);

/**
 * @brief Check if the battery level is critical
 * 
 * @return bool True if battery is critically low
 */
bool power_mgmt_is_battery_critical(void);

/**
 * @brief Get the current button state
 * 
 * @return bool True if button is pressed
 */
bool power_mgmt_get_button_state(void);

/**
 * @brief Enter deep sleep mode
 * 
 * @param sleep_time_us Sleep time in microseconds (0 for indefinite)
 * @return esp_err_t ESP_OK on success
 */
esp_err_t power_mgmt_enter_deep_sleep(uint64_t sleep_time_us);

/**
 * @brief Enter light sleep mode
 * 
 * @param sleep_time_us Sleep time in microseconds (0 for indefinite)
 * @return esp_err_t ESP_OK on success
 */
esp_err_t power_mgmt_enter_light_sleep(uint64_t sleep_time_us);

/**
 * @brief Deinitialize power management module
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t power_mgmt_deinit(void);

#endif /* POWER_MGMT_H */
