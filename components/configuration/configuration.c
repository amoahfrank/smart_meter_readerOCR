/**
 * @file configuration.c
 * @brief Configuration manager implementation for Smart Meter Reader OCR
 * 
 * This file implements the configuration manager, which handles loading, saving,
 * and managing device configuration settings.
 * 
 * @author Frank Amoah A.K.A SiRWaTT Smart Meter Reader OCR Team Lead
 * @date 2023
 */

#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "nvs.h"

// Include component headers
#include "security_manager.h"

// Include local headers
#include "configuration.h"

static const char *TAG = "configuration";

// NVS namespace and keys
#define CONFIG_NAMESPACE        "smart_meter"
#define KEY_WIFI_SSID           "wifi_ssid"
#define KEY_WIFI_PASSWORD       "wifi_pass"
#define KEY_SERVER_URL          "server_url"
#define KEY_COMM_MODE           "comm_mode"
#define KEY_READING_INTERVAL    "read_int"
#define KEY_OCR_MIN_CONFIDENCE  "ocr_conf"

// Default configuration values
#define DEFAULT_READING_INTERVAL_SEC    300     // 5 minutes
#define DEFAULT_OCR_MIN_CONFIDENCE      70      // 70%
#define DEFAULT_COMM_MODE               COMM_MODE_WIFI
#define DEFAULT_SERVER_URL              "https://example.com/api/readings"

// NVS handle
static nvs_handle_t nvs_handle;
static bool is_initialized = false;

/**
 * @brief Initialize the configuration manager
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t configuration_init(void)
{
    ESP_LOGI(TAG, "Initializing configuration manager");
    
    if (is_initialized) {
        ESP_LOGW(TAG, "Configuration manager already initialized");
        return ESP_OK;
    }
    
    // Initialize NVS
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        // NVS partition was truncated and needs to be erased
        ESP_LOGW(TAG, "NVS needs to be erased");
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(err);
    
    // Open NVS namespace
    err = nvs_open(CONFIG_NAMESPACE, NVS_READWRITE, &nvs_handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Error opening NVS namespace: %s", esp_err_to_name(err));
        return err;
    }
    
    is_initialized = true;
    ESP_LOGI(TAG, "Configuration manager initialized");
    
    return ESP_OK;
}

/**
 * @brief Set default configuration values
 * 
 * @param config Pointer to configuration structure
 * @return esp_err_t ESP_OK on success
 */
esp_err_t configuration_set_defaults(device_config_t *config)
{
    ESP_LOGI(TAG, "Setting default configuration values");
    
    if (config == NULL) {
        ESP_LOGE(TAG, "Invalid configuration pointer");
        return ESP_ERR_INVALID_ARG;
    }
    
    // Clear all fields
    memset(config, 0, sizeof(device_config_t));
    
    // Set default values
    config->reading_interval_sec = DEFAULT_READING_INTERVAL_SEC;
    config->ocr_min_confidence = DEFAULT_OCR_MIN_CONFIDENCE;
    config->comm_mode = DEFAULT_COMM_MODE;
    strncpy(config->server_url, DEFAULT_SERVER_URL, sizeof(config->server_url) - 1);
    
    ESP_LOGI(TAG, "Default configuration set");
    
    return ESP_OK;
}

/**
 * @brief Load configuration from NVS
 * 
 * @param config Pointer to store configuration
 * @return esp_err_t ESP_OK on success
 */
esp_err_t configuration_load(device_config_t *config)
{
    ESP_LOGI(TAG, "Loading configuration from NVS");
    
    if (!is_initialized) {
        ESP_LOGE(TAG, "Configuration manager not initialized");
        return ESP_ERR_INVALID_STATE;
    }
    
    if (config == NULL) {
        ESP_LOGE(TAG, "Invalid configuration pointer");
        return ESP_ERR_INVALID_ARG;
    }
    
    // Start with default values
    configuration_set_defaults(config);
    
    // Load WiFi SSID
    size_t len = sizeof(config->wifi_ssid);
    esp_err_t err = nvs_get_str(nvs_handle, KEY_WIFI_SSID, config->wifi_ssid, &len);
    if (err != ESP_OK && err != ESP_ERR_NVS_NOT_FOUND) {
        ESP_LOGE(TAG, "Error reading WiFi SSID: %s", esp_err_to_name(err));
        return err;
    }
    
    // Load WiFi password (use secure storage for sensitive data)
    err = security_manager_get_secret(KEY_WIFI_PASSWORD, config->wifi_password, sizeof(config->wifi_password));
    if (err != ESP_OK && err != ESP_ERR_NOT_FOUND) {
        ESP_LOGE(TAG, "Error reading WiFi password: %s", esp_err_to_name(err));
    }
    
    // Load server URL
    len = sizeof(config->server_url);
    err = nvs_get_str(nvs_handle, KEY_SERVER_URL, config->server_url, &len);
    if (err != ESP_OK && err != ESP_ERR_NVS_NOT_FOUND) {
        ESP_LOGE(TAG, "Error reading server URL: %s", esp_err_to_name(err));
        return err;
    }
    
    // Load communication mode
    uint8_t comm_mode;
    err = nvs_get_u8(nvs_handle, KEY_COMM_MODE, &comm_mode);
    if (err == ESP_OK) {
        config->comm_mode = (comm_mode_t)comm_mode;
    } else if (err != ESP_ERR_NVS_NOT_FOUND) {
        ESP_LOGE(TAG, "Error reading communication mode: %s", esp_err_to_name(err));
        return err;
    }
    
    // Load reading interval
    err = nvs_get_u32(nvs_handle, KEY_READING_INTERVAL, &config->reading_interval_sec);
    if (err != ESP_OK && err != ESP_ERR_NVS_NOT_FOUND) {
        ESP_LOGE(TAG, "Error reading reading interval: %s", esp_err_to_name(err));
        return err;
    }
    
    // Load OCR minimum confidence
    err = nvs_get_u8(nvs_handle, KEY_OCR_MIN_CONFIDENCE, &config->ocr_min_confidence);
    if (err != ESP_OK && err != ESP_ERR_NVS_NOT_FOUND) {
        ESP_LOGE(TAG, "Error reading OCR min confidence: %s", esp_err_to_name(err));
        return err;
    }
    
    ESP_LOGI(TAG, "Configuration loaded successfully");
    ESP_LOGI(TAG, "WiFi SSID: %s", config->wifi_ssid);
    ESP_LOGI(TAG, "Server URL: %s", config->server_url);
    ESP_LOGI(TAG, "Comm Mode: %d", config->comm_mode);
    ESP_LOGI(TAG, "Reading Interval: %d sec", config->reading_interval_sec);
    ESP_LOGI(TAG, "OCR Min Confidence: %d%%", config->ocr_min_confidence);
    
    return ESP_OK;
}

/**
 * @brief Save configuration to NVS
 * 
 * @param config Pointer to configuration to save
 * @return esp_err_t ESP_OK on success
 */
esp_err_t configuration_save(const device_config_t *config)
{
    ESP_LOGI(TAG, "Saving configuration to NVS");
    
    if (!is_initialized) {
        ESP_LOGE(TAG, "Configuration manager not initialized");
        return ESP_ERR_INVALID_STATE;
    }
    
    if (config == NULL) {
        ESP_LOGE(TAG, "Invalid configuration pointer");
        return ESP_ERR_INVALID_ARG;
    }
    
    // Save WiFi SSID
    esp_err_t err = nvs_set_str(nvs_handle, KEY_WIFI_SSID, config->wifi_ssid);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Error saving WiFi SSID: %s", esp_err_to_name(err));
        return err;
    }
    
    // Save WiFi password (use secure storage for sensitive data)
    err = security_manager_set_secret(KEY_WIFI_PASSWORD, config->wifi_password, strlen(config->wifi_password));
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Error saving WiFi password: %s", esp_err_to_name(err));
        return err;
    }
    
    // Save server URL
    err = nvs_set_str(nvs_handle, KEY_SERVER_URL, config->server_url);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Error saving server URL: %s", esp_err_to_name(err));
        return err;
    }
    
    // Save communication mode
    err = nvs_set_u8(nvs_handle, KEY_COMM_MODE, (uint8_t)config->comm_mode);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Error saving communication mode: %s", esp_err_to_name(err));
        return err;
    }
    
    // Save reading interval
    err = nvs_set_u32(nvs_handle, KEY_READING_INTERVAL, config->reading_interval_sec);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Error saving reading interval: %s", esp_err_to_name(err));
        return err;
    }
    
    // Save OCR minimum confidence
    err = nvs_set_u8(nvs_handle, KEY_OCR_MIN_CONFIDENCE, config->ocr_min_confidence);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Error saving OCR min confidence: %s", esp_err_to_name(err));
        return err;
    }
    
    // Commit changes
    err = nvs_commit(nvs_handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Error committing NVS changes: %s", esp_err_to_name(err));
        return err;
    }
    
    ESP_LOGI(TAG, "Configuration saved successfully");
    
    return ESP_OK;
}

/**
 * @brief Deinitialize the configuration manager
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t configuration_deinit(void)
{
    ESP_LOGI(TAG, "Deinitializing configuration manager");
    
    if (!is_initialized) {
        ESP_LOGW(TAG, "Configuration manager not initialized");
        return ESP_OK;
    }
    
    // Close NVS handle
    nvs_close(nvs_handle);
    
    is_initialized = false;
    ESP_LOGI(TAG, "Configuration manager deinitialized");
    
    return ESP_OK;
}
