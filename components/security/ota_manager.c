
/**
 * @file ota_manager.c
 * @brief OTA update manager implementation for Smart Meter Reader OCR
 * 
 * This file implements the OTA (Over-The-Air) update functionality,
 * including firmware verification, download, and application.
 * 
 * @author Smart Meter Reader OCR Team
 * @date 2023
 */

#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_ota_ops.h"
#include "esp_http_client.h"
#include "esp_https_ota.h"
#include "esp_app_format.h"
#include "nvs_flash.h"
#include "nvs.h"

// Include component headers
#include "security_manager.h"
#include "configuration.h"

// Include local headers
#include "ota_manager.h"

static const char *TAG = "ota_manager";

// OTA configuration
#define OTA_BUFFER_SIZE         1024
#define OTA_FIRMWARE_UPGRADE_URL_SIZE 256

// NVS keys
#define NVS_OTA_NAMESPACE       "ota_manager"
#define NVS_FIRMWARE_VERSION    "fw_version"
#define NVS_LAST_CHECK_TIME     "last_check"

// OTA update interval (24 hours)
#define OTA_CHECK_INTERVAL_SEC  (24 * 60 * 60)

// Current firmware version
#define FIRMWARE_VERSION        "1.0.0"

// Default OTA server URL
#ifndef CONFIG_OTA_SERVER_URL
#define CONFIG_OTA_SERVER_URL   "https://example.com/firmware"
#endif

// OTA internal state
static bool is_initialized = false;
static nvs_handle_t nvs_handle;
static char ota_server_url[OTA_FIRMWARE_UPGRADE_URL_SIZE] = CONFIG_OTA_SERVER_URL;

// Forward declarations
static esp_err_t http_event_handler(esp_http_client_event_t *evt);
static esp_err_t validate_image_header(esp_app_desc_t *new_app_info);
static void ota_task(void *pvParameter);

/**
 * @brief Initialize the OTA manager
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t ota_manager_init(void)
{
    ESP_LOGI(TAG, "Initializing OTA manager");
    
    if (is_initialized) {
        ESP_LOGW(TAG, "OTA manager already initialized");
        return ESP_OK;
    }
    
    // Initialize NVS if not already initialized
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        // NVS partition was truncated and needs to be erased
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(err);
    
    // Open NVS namespace
    err = nvs_open(NVS_OTA_NAMESPACE, NVS_READWRITE, &nvs_handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Error opening NVS namespace: %s", esp_err_to_name(err));
        return err;
    }
    
    // Store current firmware version if not already stored
    char stored_version[32] = {0};
    size_t version_len = sizeof(stored_version);
    
    err = nvs_get_str(nvs_handle, NVS_FIRMWARE_VERSION, stored_version, &version_len);
    if (err == ESP_ERR_NVS_NOT_FOUND) {
        // Store current firmware version
        err = nvs_set_str(nvs_handle, NVS_FIRMWARE_VERSION, FIRMWARE_VERSION);
        if (err != ESP_OK) {
            ESP_LOGE(TAG, "Failed to store firmware version: %s", esp_err_to_name(err));
            return err;
        }
        
        // Commit changes
        err = nvs_commit(nvs_handle);
        if (err != ESP_OK) {
            ESP_LOGE(TAG, "Failed to commit NVS changes: %s", esp_err_to_name(err));
            return err;
        }
    } else if (err != ESP_OK) {
        ESP_LOGE(TAG, "Error reading firmware version: %s", esp_err_to_name(err));
        return err;
    }
    
    // Log current firmware version
    const esp_app_desc_t *app_desc = esp_app_get_description();
    ESP_LOGI(TAG, "Current firmware version: %s", app_desc->version);
    
    is_initialized = true;
    ESP_LOGI(TAG, "OTA manager initialized");
    
    return ESP_OK;
}

/**
 * @brief Set the OTA server URL
 * 
 * @param url The OTA server URL
 * @return esp_err_t ESP_OK on success
 */
esp_err_t ota_manager_set_server_url(const char *url)
{
    ESP_LOGI(TAG, "Setting OTA server URL: %s", url);
    
    if (!is_initialized) {
        ESP_LOGE(TAG, "OTA manager not initialized");
        return ESP_ERR_INVALID_STATE;
    }
    
    if (url == NULL) {
        ESP_LOGE(TAG, "Invalid URL");
        return ESP_ERR_INVALID_ARG;
    }
    
    if (strlen(url) >= OTA_FIRMWARE_UPGRADE_URL_SIZE) {
        ESP_LOGE(TAG, "URL too long");
        return ESP_ERR_INVALID_ARG;
    }
    
    // Copy URL
    strncpy(ota_server_url, url, OTA_FIRMWARE_UPGRADE_URL_SIZE - 1);
    ota_server_url[OTA_FIRMWARE_UPGRADE_URL_SIZE - 1] = '\0';
    
    ESP_LOGI(TAG, "OTA server URL set successfully");
    
    return ESP_OK;
}

/**
 * @brief Check if an OTA update is available
 * 
 * @return bool True if update is available
 */
bool ota_manager_check_update(void)
{
    ESP_LOGI(TAG, "Checking for OTA updates");
    
    if (!is_initialized) {
        ESP_LOGE(TAG, "OTA manager not initialized");
        return false;
    }
    
    // Check if it's too soon to check for updates
    int64_t last_check_time = 0;
    esp_err_t err = nvs_get_i64(nvs_handle, NVS_LAST_CHECK_TIME, &last_check_time);
    
    if (err == ESP_OK) {
        int64_t current_time = time(NULL);
        int64_t elapsed_time = current_time - last_check_time;
        
        // Skip check if last check was recent (unless this is a forced check)
        if (elapsed_time < OTA_CHECK_INTERVAL_SEC) {
            ESP_LOGI(TAG, "Skipping OTA check (checked %lld seconds ago)", elapsed_time);
            return false;
        }
    } else if (err != ESP_ERR_NVS_NOT_FOUND) {
        ESP_LOGW(TAG, "Failed to get last check time: %s", esp_err_to_name(err));
    }
    
    // Configure HTTP client for update check
    esp_http_client_config_t http_config = {
        .url = ota_server_url,
        .event_handler = http_event_handler,
        .timeout_ms = 5000,
        .buffer_size = OTA_BUFFER_SIZE,
        .skip_cert_common_name_check = false,
    };
    
    // Configure TLS
    esp_tls_cfg_t tls_cfg;
    security_manager_configure_tls(&tls_cfg);
    http_config.crt_bundle_attach = tls_cfg.crt_bundle_attach;
    
    // Create HTTP client
    esp_http_client_handle_t client = esp_http_client_init(&http_config);
    if (client == NULL) {
        ESP_LOGE(TAG, "Failed to initialize HTTP client");
        return false;
    }
    
    // Set headers
    esp_http_client_set_header(client, "Accept", "application/json");
    
    // Add authentication (if configured)
    char auth_header[128];
    if (security_manager_get_auth_header(auth_header, sizeof(auth_header)) == ESP_OK) {
        esp_http_client_set_header(client, "Authorization", auth_header);
    }
    
    // Set method to HEAD to only fetch headers
    esp_http_client_set_method(client, HTTP_METHOD_HEAD);
    
    // Perform HTTP request
    esp_err_t ret = esp_http_client_perform(client);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "HTTP request failed: %s", esp_err_to_name(ret));
        esp_http_client_cleanup(client);
        return false;
    }
    
    // Get status code
    int status_code = esp_http_client_get_status_code(client);
    if (status_code != 200) {
        ESP_LOGW(TAG, "HTTP request returned status %d", status_code);
        esp_http_client_cleanup(client);
        return false;
    }
    
    // Get firmware version from header
    char fw_version[32] = {0};
    esp_http_client_get_header(client, "X-Firmware-Version", fw_version, sizeof(fw_version));
    
    // Get current app description
    const esp_app_desc_t *app_desc = esp_app_get_description();
    
    ESP_LOGI(TAG, "Current firmware version: %s", app_desc->version);
    ESP_LOGI(TAG, "Available firmware version: %s", fw_version);
    
    // Check if new version is available
    bool update_available = (strlen(fw_version) > 0 && strcmp(fw_version, app_desc->version) != 0);
    
    // Clean up HTTP client
    esp_http_client_cleanup(client);
    
    // Update last check time
    int64_t current_time = time(NULL);
    nvs_set_i64(nvs_handle, NVS_LAST_CHECK_TIME, current_time);
    nvs_commit(nvs_handle);
    
    ESP_LOGI(TAG, "OTA update %s", update_available ? "available" : "not available");
    
    return update_available;
}

/**
 * @brief Perform OTA update
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t ota_manager_perform_update(void)
{
    ESP_LOGI(TAG, "Starting OTA update");
    
    if (!is_initialized) {
        ESP_LOGE(TAG, "OTA manager not initialized");
        return ESP_ERR_INVALID_STATE;
    }
    
    // Create OTA update task
    TaskHandle_t task_handle = NULL;
    BaseType_t task_created = xTaskCreate(
        ota_task,
        "ota_task",
        8192,
        NULL,
        5,
        &task_handle
    );
    
    if (task_created != pdPASS) {
        ESP_LOGE(TAG, "Failed to create OTA task");
        return ESP_FAIL;
    }
    
    // Wait for task to complete
    // This is a simple approach; in a real application you might want to use event groups
    // for better task synchronization
    vTaskDelay(pdMS_TO_TICKS(1000));
    
    // The OTA task will restart the device on success, so if we get here, it failed
    if (eTaskGetState(task_handle) != eDeleted) {
        vTaskDelete(task_handle);
        ESP_LOGE(TAG, "OTA update failed");
        return ESP_FAIL;
    }
    
    return ESP_OK;
}

/**
 * @brief OTA update task
 * 
 * @param pvParameter Task parameters (unused)
 */
static void ota_task(void *pvParameter)
{
    ESP_LOGI(TAG, "OTA task started");
    
    // Configure HTTP client for OTA update
    esp_http_client_config_t http_config = {
        .url = ota_server_url,
        .event_handler = http_event_handler,
        .timeout_ms = 10000,
        .buffer_size = OTA_BUFFER_SIZE,
        .skip_cert_common_name_check = false,
    };
    
    // Configure TLS
    esp_tls_cfg_t tls_cfg;
    security_manager_configure_tls(&tls_cfg);
    http_config.crt_bundle_attach = tls_cfg.crt_bundle_attach;
    
    // Configure OTA function
    esp_https_ota_config_t ota_config = {
        .http_config = &http_config,
    };
    
    // Add authentication (if configured)
    char auth_header[128];
    if (security_manager_get_auth_header(auth_header, sizeof(auth_header)) == ESP_OK) {
        http_config.auth_type = HTTP_AUTH_TYPE_BASIC;
        http_config.username = auth_header; // Used as auth header in this case
    }
    
    ESP_LOGI(TAG, "Starting OTA update from %s", ota_server_url);
    
    // Start OTA update
    esp_err_t ret = esp_https_ota(&ota_config);
    
    if (ret == ESP_OK) {
        ESP_LOGI(TAG, "OTA update successful");
        
        // Update firmware version in NVS
        const esp_app_desc_t *app_desc = esp_app_get_description();
        nvs_set_str(nvs_handle, NVS_FIRMWARE_VERSION, app_desc->version);
        nvs_commit(nvs_handle);
        
        ESP_LOGI(TAG, "Restarting system...");
        vTaskDelay(pdMS_TO_TICKS(1000));
        esp_restart();
    } else {
        ESP_LOGE(TAG, "OTA update failed: %s", esp_err_to_name(ret));
    }
    
    vTaskDelete(NULL);
}

/**
 * @brief HTTP client event handler
 * 
 * @param evt HTTP client event
 * @return esp_err_t ESP_OK on success
 */
static esp_err_t http_event_handler(esp_http_client_event_t *evt)
{
    switch (evt->event_id) {
        case HTTP_EVENT_ERROR:
            ESP_LOGW(TAG, "HTTP client error");
            break;
        case HTTP_EVENT_ON_CONNECTED:
            ESP_LOGI(TAG, "HTTP client connected");
            break;
        case HTTP_EVENT_HEADERS_SENT:
            ESP_LOGI(TAG, "HTTP headers sent");
            break;
        case HTTP_EVENT_ON_HEADER:
            ESP_LOGD(TAG, "HTTP header received: %s: %s", evt->header_key, evt->header_value);
            break;
        case HTTP_EVENT_ON_DATA:
            ESP_LOGD(TAG, "HTTP data received (%d bytes)", evt->data_len);
            break;
        case HTTP_EVENT_ON_FINISH:
            ESP_LOGI(TAG, "HTTP request finished");
            break;
        case HTTP_EVENT_DISCONNECTED:
            ESP_LOGI(TAG, "HTTP client disconnected");
            break;
        default:
            break;
    }
    return ESP_OK;
}

/**
 * @brief Validate new firmware image header
 * 
 * @param new_app_info New firmware app description
 * @return esp_err_t ESP_OK if valid
 */
static esp_err_t validate_image_header(esp_app_desc_t *new_app_info)
{
    if (new_app_info == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    
    ESP_LOGI(TAG, "Validating firmware image header");
    
    // Get current app description
    const esp_app_desc_t *running_app_info = esp_app_get_description();
    
    // Check project name
    if (strcmp(new_app_info->project_name, running_app_info->project_name) != 0) {
        ESP_LOGW(TAG, "Project name mismatch, expected %s, got %s",
                 running_app_info->project_name, new_app_info->project_name);
        return ESP_FAIL;
    }
    
    // Check version (allow downgrade for testing, but warn about it)
    if (strcmp(new_app_info->version, running_app_info->version) < 0) {
        ESP_LOGW(TAG, "Firmware downgrade detected: %s -> %s",
                 running_app_info->version, new_app_info->version);
        // In production, you might want to return ESP_FAIL here to prevent downgrades
    }
    
    // In a production environment, you would also want to verify the
    // firmware signature here using secure boot capabilities
    
    ESP_LOGI(TAG, "Firmware image header valid");
    
    return ESP_OK;
}

/**
 * @brief Deinitialize the OTA manager
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t ota_manager_deinit(void)
{
    ESP_LOGI(TAG, "Deinitializing OTA manager");
    
    if (!is_initialized) {
        ESP_LOGW(TAG, "OTA manager not initialized");
        return ESP_OK;
    }
    
    // Close NVS handle
    nvs_close(nvs_handle);
    
    is_initialized = false;
    ESP_LOGI(TAG, "OTA manager deinitialized");
    
    return ESP_OK;
}
