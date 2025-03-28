/**
 * @file app_main.c
 * @brief Main application file for Smart Meter Reader OCR
 * 
 * This file contains the entry point and main initialization for the
 * Smart Meter Reader OCR device. It initializes all subsystems and
 * starts the main application tasks.
 * 
 * @author Smart Meter Reader OCR Team
 * @date 2023
 */

#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "esp_system.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "esp_sleep.h"

// Include component headers
#include "camera.h"
#include "display.h"
#include "ocr.h"
#include "wifi_manager.h"
#include "ble_manager.h"
#include "power_mgmt.h"
#include "security_manager.h"
#include "ota_manager.h"
#include "configuration.h"
#include "state_machine.h"

// Local includes
#include "app_main.h"

static const char *TAG = "app_main";

// Event group to signal between tasks
EventGroupHandle_t event_group;

// System state
system_state_t system_state = {
    .current_state = STATE_INIT,
    .battery_level = 0,
    .last_reading = {0},
    .error_state = ERROR_NONE,
    .config = {0}
};

// Forward declarations
static void initialize_nvs(void);
static void initialize_components(void);
static void print_system_info(void);

/**
 * @brief Application entry point
 */
void app_main(void)
{
    ESP_LOGI(TAG, "Starting Smart Meter Reader OCR");
    
    // Create event group
    event_group = xEventGroupCreate();
    
    // Initialize NVS (Non-volatile storage)
    initialize_nvs();
    
    // Initialize security manager (must be early in boot process)
    security_manager_init();
    
    // Print system information
    print_system_info();
    
    // Check wakeup cause
    esp_sleep_wakeup_cause_t wakeup_cause = esp_sleep_get_wakeup_cause();
    if (wakeup_cause == ESP_SLEEP_WAKEUP_TIMER) {
        ESP_LOGI(TAG, "Wakeup caused by timer");
        system_state.current_state = STATE_CAPTURE;
    } else {
        ESP_LOGI(TAG, "Normal boot");
        // Initialize all hardware components
        initialize_components();
    }
    
    // Load configuration
    if (configuration_load(&system_state.config) != ESP_OK) {
        ESP_LOGW(TAG, "Failed to load configuration, using defaults");
        configuration_set_defaults(&system_state.config);
        configuration_save(&system_state.config);
    }
    
    // Check if button is pressed during boot (enter config mode)
    if (power_mgmt_get_button_state()) {
        system_state.current_state = STATE_CONFIG;
        ESP_LOGI(TAG, "Button pressed during boot, entering configuration mode");
    }
    
    // Start the state machine
    state_machine_start(&system_state, event_group);
    
    // The main task ends here, but the state machine keeps running
    ESP_LOGI(TAG, "Main initialization complete, system running");
}

/**
 * @brief Initialize NVS (Non-volatile storage)
 */
static void initialize_nvs(void)
{
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_LOGW(TAG, "NVS partition needs to be erased");
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);
    ESP_LOGI(TAG, "NVS initialized");
}

/**
 * @brief Initialize all system components
 */
static void initialize_components(void)
{
    ESP_LOGI(TAG, "Initializing system components");
    
    // Initialize power management (early for accurate battery readings)
    power_mgmt_init();
    system_state.battery_level = power_mgmt_get_battery_level();
    ESP_LOGI(TAG, "Battery level: %d%%", system_state.battery_level);
    
    // Initialize configuration manager
    configuration_init();
    
    // Initialize camera
    camera_init_config_t camera_config = {
        .xclk_freq_hz = CONFIG_XCLK_FREQ,
        .pixel_format = CAMERA_PIXEL_FORMAT,
        .frame_size = CAMERA_FRAME_SIZE,
        .jpeg_quality = 12,
        .fb_count = 1
    };
    ESP_ERROR_CHECK(camera_init(&camera_config));
    
    // Initialize display
    display_init();
    display_clear();
    display_show_splash();
    
    // Initialize OCR engine
    ocr_init();
    
    // Initialize wireless connectivity
    wifi_manager_init();
    ble_manager_init(event_group);
    
    // Initialize OTA manager
    ota_manager_init();
    
    ESP_LOGI(TAG, "All components initialized");
}

/**
 * @brief Print system information
 */
static void print_system_info(void)
{
    esp_chip_info_t chip_info;
    esp_chip_info(&chip_info);
    
    ESP_LOGI(TAG, "Smart Meter Reader OCR");
    ESP_LOGI(TAG, "IDF Version: %s", esp_get_idf_version());
    ESP_LOGI(TAG, "Chip info:");
    ESP_LOGI(TAG, " - model: %s", CONFIG_IDF_TARGET);
    ESP_LOGI(TAG, " - cores: %d", chip_info.cores);
    ESP_LOGI(TAG, " - feature: %s%s%s%s%s",
             chip_info.features & CHIP_FEATURE_WIFI_BGN ? "WiFi " : "",
             chip_info.features & CHIP_FEATURE_BT ? "BT " : "",
             chip_info.features & CHIP_FEATURE_BLE ? "BLE " : "",
             chip_info.features & CHIP_FEATURE_IEEE802154 ? "802.15.4 " : "",
             "");
    ESP_LOGI(TAG, " - revision number: %d", chip_info.revision);
    
    uint8_t mac[6];
    esp_efuse_mac_get_default(mac);
    ESP_LOGI(TAG, "MAC: %02X:%02X:%02X:%02X:%02X:%02X", 
             mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
}
