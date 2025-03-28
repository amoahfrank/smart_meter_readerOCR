/**
 * @file power_mgmt.c
 * @brief Power management implementation for Smart Meter Reader OCR
 * 
 * This file implements power management functionality, including battery monitoring,
 * power modes, and sleep functionality to optimize battery life.
 * 
 * @author Smart Meter Reader OCR Team
 * @date 2023
 */

#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_sleep.h"
#include "esp_adc/adc_oneshot.h"
#include "esp_adc/adc_cali.h"
#include "esp_adc/adc_cali_scheme.h"
#include "driver/gpio.h"

// Include local headers
#include "power_mgmt.h"

static const char *TAG = "power_mgmt";

// ADC channel for battery voltage monitoring
#define BATTERY_ADC_CHANNEL       ADC_CHANNEL_6  // GPIO34 on most ESP32 boards
#define BUTTON_GPIO              GPIO_NUM_0     // GPIO0 on most ESP32 boards
#define BATTERY_ADC_ATTEN        ADC_ATTEN_DB_11
#define BATTERY_ADC_UNIT         ADC_UNIT_1

// Battery voltage thresholds (in mV)
#define BATTERY_FULL_MV          4200
#define BATTERY_EMPTY_MV         3300
#define BATTERY_CRITICAL_MV      3400
#define BATTERY_VOLTAGE_DIVIDER  2.0f  // Voltage divider ratio, if used

// ADC calibration handle
static bool adc_calibrated = false;
static adc_cali_handle_t adc_cali_handle = NULL;
static adc_oneshot_unit_handle_t adc_handle;

// Button configuration
static bool button_initialized = false;

/**
 * @brief Initialize the power management module
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t power_mgmt_init(void)
{
    ESP_LOGI(TAG, "Initializing power management");
    
    // Initialize ADC for battery monitoring
    adc_oneshot_unit_init_cfg_t init_config = {
        .unit_id = BATTERY_ADC_UNIT,
    };
    ESP_ERROR_CHECK(adc_oneshot_new_unit(&init_config, &adc_handle));
    
    // Configure ADC channel
    adc_oneshot_chan_cfg_t config = {
        .bitwidth = ADC_BITWIDTH_DEFAULT,
        .atten = BATTERY_ADC_ATTEN,
    };
    ESP_ERROR_CHECK(adc_oneshot_config_channel(adc_handle, BATTERY_ADC_CHANNEL, &config));
    
    // Characterize ADC
    adc_cali_curve_fitting_config_t cali_config = {
        .unit_id = BATTERY_ADC_UNIT,
        .atten = BATTERY_ADC_ATTEN,
        .bitwidth = ADC_BITWIDTH_DEFAULT,
    };
    
    esp_err_t ret = adc_cali_create_scheme_curve_fitting(&cali_config, &adc_cali_handle);
    if (ret == ESP_OK) {
        adc_calibrated = true;
    } else {
        ESP_LOGW(TAG, "ADC calibration failed, using uncalibrated readings");
    }
    
    // Configure button for wakeup and user input
    gpio_config_t io_conf = {
        .pin_bit_mask = (1ULL << BUTTON_GPIO),
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    gpio_config(&io_conf);
    button_initialized = true;
    
    // Enable wakeup from deep sleep via GPIO
    esp_sleep_enable_ext0_wakeup(BUTTON_GPIO, 0); // 0 = LOW level to trigger wakeup
    
    ESP_LOGI(TAG, "Power management initialized");
    
    return ESP_OK;
}

/**
 * @brief Get the current battery level as a percentage
 * 
 * @return int Battery level (0-100)
 */
int power_mgmt_get_battery_level(void)
{
    int battery_voltage = power_mgmt_get_battery_voltage();
    
    // Calculate percentage
    int battery_range = BATTERY_FULL_MV - BATTERY_EMPTY_MV;
    int voltage_offset = battery_voltage - BATTERY_EMPTY_MV;
    
    // Clamp to valid range
    if (voltage_offset <= 0) {
        return 0;
    }
    if (voltage_offset >= battery_range) {
        return 100;
    }
    
    // Convert to percentage
    int percentage = (voltage_offset * 100) / battery_range;
    
    ESP_LOGI(TAG, "Battery level: %d%% (%d mV)", percentage, battery_voltage);
    
    return percentage;
}

/**
 * @brief Get the current battery voltage in millivolts
 * 
 * @return int Battery voltage (mV)
 */
int power_mgmt_get_battery_voltage(void)
{
    int adc_raw = 0;
    int voltage = 0;
    
    // Read raw ADC value
    ESP_ERROR_CHECK(adc_oneshot_read(adc_handle, BATTERY_ADC_CHANNEL, &adc_raw));
    
    // Convert to voltage
    if (adc_calibrated) {
        ESP_ERROR_CHECK(adc_cali_raw_to_voltage(adc_cali_handle, adc_raw, &voltage));
    } else {
        // Approximate using formula (adjust for your hardware)
        voltage = (adc_raw * 3300) / 4095;
    }
    
    // Apply voltage divider correction (if used)
    voltage = (int)(voltage * BATTERY_VOLTAGE_DIVIDER);
    
    ESP_LOGI(TAG, "Battery voltage: %d mV (ADC raw: %d)", voltage, adc_raw);
    
    return voltage;
}

/**
 * @brief Check if the battery level is critical
 * 
 * @return bool True if battery is critically low
 */
bool power_mgmt_is_battery_critical(void)
{
    int battery_voltage = power_mgmt_get_battery_voltage();
    
    if (battery_voltage <= BATTERY_CRITICAL_MV) {
        ESP_LOGW(TAG, "Battery level is critical: %d mV", battery_voltage);
        return true;
    }
    
    return false;
}

/**
 * @brief Get the current button state
 * 
 * @return bool True if button is pressed
 */
bool power_mgmt_get_button_state(void)
{
    if (!button_initialized) {
        ESP_LOGE(TAG, "Button not initialized");
        return false;
    }
    
    // Read button state (inverted because of pull-up)
    bool pressed = !gpio_get_level(BUTTON_GPIO);
    
    return pressed;
}

/**
 * @brief Enter deep sleep mode
 * 
 * @param sleep_time_us Sleep time in microseconds (0 for indefinite)
 * @return esp_err_t ESP_OK on success
 */
esp_err_t power_mgmt_enter_deep_sleep(uint64_t sleep_time_us)
{
    ESP_LOGI(TAG, "Entering deep sleep for %llu microseconds", sleep_time_us);
    
    // Configure sleep time if specified
    if (sleep_time_us > 0) {
        esp_sleep_enable_timer_wakeup(sleep_time_us);
    }
    
    // Enter deep sleep
    esp_deep_sleep_start();
    
    // Execution will never reach here
    return ESP_OK;
}

/**
 * @brief Enter light sleep mode
 * 
 * @param sleep_time_us Sleep time in microseconds (0 for indefinite)
 * @return esp_err_t ESP_OK on success
 */
esp_err_t power_mgmt_enter_light_sleep(uint64_t sleep_time_us)
{
    ESP_LOGI(TAG, "Entering light sleep for %llu microseconds", sleep_time_us);
    
    // Configure sleep time if specified
    if (sleep_time_us > 0) {
        esp_sleep_enable_timer_wakeup(sleep_time_us);
    }
    
    // Enter light sleep
    esp_light_sleep_start();
    
    // Execution continues after wakeup
    esp_sleep_wakeup_cause_t wakeup_reason = esp_sleep_get_wakeup_cause();
    
    ESP_LOGI(TAG, "Woke up from light sleep, reason: %d", wakeup_reason);
    
    return ESP_OK;
}

/**
 * @brief Deinitialize power management module
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t power_mgmt_deinit(void)
{
    ESP_LOGI(TAG, "Deinitializing power management");
    
    // Delete ADC unit
    ESP_ERROR_CHECK(adc_oneshot_del_unit(adc_handle));
    
    // Delete calibration handle if created
    if (adc_calibrated) {
        ESP_ERROR_CHECK(adc_cali_delete_scheme_curve_fitting(adc_cali_handle));
        adc_calibrated = false;
    }
    
    ESP_LOGI(TAG, "Power management deinitialized");
    
    return ESP_OK;
}
