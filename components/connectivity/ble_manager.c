/**
 * @file ble_manager.c
 * @brief BLE connectivity manager implementation
 * 
 * This file implements the BLE connectivity manager for the Smart Meter Reader.
 * It handles BLE initialization, advertising, service creation, configuration,
 * and data transmission.
 * 
 * @author Frank Amoah A.K.A SiRWaTT Smart Meter Reader OCR Team Lead
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
#include "esp_bt.h"
#include "esp_gap_ble_api.h"
#include "esp_gatts_api.h"
#include "esp_bt_main.h"
#include "esp_bt_device.h"

// Include component headers
#include "configuration.h"

// Include local headers
#include "ble_manager.h"

static const char *TAG = "ble_manager";

// BLE configuration
#define DEVICE_NAME             "Smart Meter Reader"
#define MANUFACTURER_DATA_LEN   4
#define SERVICE_UUID_SIZE       16
#define CHAR_UUID_SIZE          16

// BLE service and characteristic UUIDs
static uint8_t service_uuid[SERVICE_UUID_SIZE] = {
    /* LSB <--------------------------------------------------------------------------------> MSB */
    0xfb, 0x34, 0x9b, 0x5f, 0x80, 0x00, 0x00, 0x80, 0x00, 0x10, 0x00, 0x00, 0xFF, 0x00, 0x00, 0x00
};

static uint8_t char_data_uuid[CHAR_UUID_SIZE] = {
    /* LSB <--------------------------------------------------------------------------------> MSB */
    0xfb, 0x34, 0x9b, 0x5f, 0x80, 0x00, 0x00, 0x80, 0x00, 0x10, 0x00, 0x00, 0xAA, 0x00, 0x00, 0x00
};

static uint8_t char_config_uuid[CHAR_UUID_SIZE] = {
    /* LSB <--------------------------------------------------------------------------------> MSB */
    0xfb, 0x34, 0x9b, 0x5f, 0x80, 0x00, 0x00, 0x80, 0x00, 0x10, 0x00, 0x00, 0xBB, 0x00, 0x00, 0x00
};

// BLE profile and handle declarations
static uint8_t adv_config_done = 0;
#define ADV_CONFIG_FLAG     (1 << 0)
#define SCAN_RSP_CONFIG_FLAG    (1 << 1)

static uint16_t service_handle;
static uint16_t char_data_handle;
static uint16_t char_config_handle;

// Current device state
static bool ble_initialized = false;
static bool in_config_mode = false;

// Reference to event group
static EventGroupHandle_t event_group = NULL;
static system_state_t *system_state = NULL;

// Forward declarations
static void gatts_profile_event_handler(esp_gatts_cb_event_t event, esp_gatt_if_t gatts_if, 
                                      esp_ble_gatts_cb_param_t *param);
static void gap_event_handler(esp_gap_ble_cb_event_t event, esp_ble_gap_cb_param_t *param);
static void gatts_event_handler(esp_gatts_cb_event_t event, esp_gatt_if_t gatts_if, 
                              esp_ble_gatts_cb_param_t *param);
static void process_config_command(const char *cmd, uint16_t len);

/**
 * @brief Initialize the BLE manager
 * 
 * @param evt_group Event group for signaling
 * @return esp_err_t ESP_OK on success
 */
esp_err_t ble_manager_init(EventGroupHandle_t evt_group)
{
    ESP_LOGI(TAG, "Initializing BLE manager");
    
    if (ble_initialized) {
        ESP_LOGW(TAG, "BLE already initialized");
        return ESP_OK;
    }
    
    // Store event group
    event_group = evt_group;
    
    // Initialize NVS (if not already done)
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);
    
    // Initialize Bluetooth controller
    esp_bt_controller_config_t bt_cfg = BT_CONTROLLER_INIT_CONFIG_DEFAULT();
    ret = esp_bt_controller_init(&bt_cfg);
    if (ret) {
        ESP_LOGE(TAG, "Failed to initialize BT controller: %s", esp_err_to_name(ret));
        return ret;
    }
    
    // Enable Bluetooth controller in BLE mode
    ret = esp_bt_controller_enable(ESP_BT_MODE_BLE);
    if (ret) {
        ESP_LOGE(TAG, "Failed to enable BT controller: %s", esp_err_to_name(ret));
        return ret;
    }
    
    // Initialize Bluedroid stack
    ret = esp_bluedroid_init();
    if (ret) {
        ESP_LOGE(TAG, "Failed to initialize Bluedroid: %s", esp_err_to_name(ret));
        return ret;
    }
    
    // Enable Bluedroid stack
    ret = esp_bluedroid_enable();
    if (ret) {
        ESP_LOGE(TAG, "Failed to enable Bluedroid: %s", esp_err_to_name(ret));
        return ret;
    }
    
    // Register GAP callback
    ret = esp_ble_gap_register_callback(gap_event_handler);
    if (ret) {
        ESP_LOGE(TAG, "Failed to register GAP callback: %s", esp_err_to_name(ret));
        return ret;
    }
    
    // Register GATTS callback
    ret = esp_ble_gatts_register_callback(gatts_event_handler);
    if (ret) {
        ESP_LOGE(TAG, "Failed to register GATTS callback: %s", esp_err_to_name(ret));
        return ret;
    }
    
    // Register GATT application
    ret = esp_ble_gatts_app_register(0);
    if (ret) {
        ESP_LOGE(TAG, "Failed to register GATT application: %s", esp_err_to_name(ret));
        return ret;
    }
    
    // Set device name
    ret = esp_ble_gap_set_device_name(DEVICE_NAME);
    if (ret) {
        ESP_LOGE(TAG, "Failed to set device name: %s", esp_err_to_name(ret));
        return ret;
    }
    
    // Configure advertising data
    esp_ble_adv_data_t adv_data = {
        .set_scan_rsp = false,
        .include_name = true,
        .include_txpower = true,
        .min_interval = 0x0006, // 7.5ms
        .max_interval = 0x0010, // 20ms
        .appearance = 0x00,
        .manufacturer_len = MANUFACTURER_DATA_LEN,
        .p_manufacturer_data = (uint8_t[]){0x01, 0x02, 0x03, 0x04},
        .service_data_len = 0,
        .p_service_data = NULL,
        .service_uuid_len = sizeof(service_uuid),
        .p_service_uuid = service_uuid,
        .flag = (ESP_BLE_ADV_FLAG_GEN_DISC | ESP_BLE_ADV_FLAG_BREDR_NOT_SPT),
    };
    
    // Configure scan response data
    esp_ble_adv_data_t scan_rsp_data = {
        .set_scan_rsp = true,
        .include_name = true,
        .include_txpower = true,
        .appearance = 0x00,
        .manufacturer_len = 0,
        .p_manufacturer_data = NULL,
        .service_data_len = 0,
        .p_service_data = NULL,
        .service_uuid_len = sizeof(service_uuid),
        .p_service_uuid = service_uuid,
        .flag = (ESP_BLE_ADV_FLAG_GEN_DISC | ESP_BLE_ADV_FLAG_BREDR_NOT_SPT),
    };
    
    // Set advertising data
    ret = esp_ble_gap_config_adv_data(&adv_data);
    if (ret) {
        ESP_LOGE(TAG, "Failed to configure advertising data: %s", esp_err_to_name(ret));
        return ret;
    }
    adv_config_done |= ADV_CONFIG_FLAG;
    
    // Set scan response data
    ret = esp_ble_gap_config_adv_data(&scan_rsp_data);
    if (ret) {
        ESP_LOGE(TAG, "Failed to configure scan response data: %s", esp_err_to_name(ret));
        return ret;
    }
    adv_config_done |= SCAN_RSP_CONFIG_FLAG;
    
    // Configure advertising parameters
    esp_ble_adv_params_t adv_params = {
        .adv_int_min = 0x20,  // 20ms
        .adv_int_max = 0x40,  // 40ms
        .adv_type = ADV_TYPE_IND,
        .own_addr_type = BLE_ADDR_TYPE_PUBLIC,
        .channel_map = ADV_CHNL_ALL,
        .adv_filter_policy = ADV_FILTER_ALLOW_SCAN_ANY_CON_ANY,
    };
    
    // Set advertising parameters
    ret = esp_ble_gap_start_advertising(&adv_params);
    if (ret) {
        ESP_LOGE(TAG, "Failed to start advertising: %s", esp_err_to_name(ret));
        return ret;
    }
    
    ble_initialized = true;
    ESP_LOGI(TAG, "BLE manager initialized successfully");
    
    return ESP_OK;
}

/**
 * @brief Start configuration mode via BLE
 * 
 * @param state System state structure
 * @return esp_err_t ESP_OK on success
 */
esp_err_t ble_manager_start_config_mode(system_state_t *state)
{
    ESP_LOGI(TAG, "Starting BLE configuration mode");
    
    if (!ble_initialized) {
        ESP_LOGE(TAG, "BLE not initialized");
        return ESP_ERR_INVALID_STATE;
    }
    
    if (in_config_mode) {
        ESP_LOGW(TAG, "Already in configuration mode");
        return ESP_OK;
    }
    
    // Store system state
    system_state = state;
    
    // Update device name to indicate config mode
    char config_name[32];
    snprintf(config_name, sizeof(config_name), "%s (Config)", DEVICE_NAME);
    esp_err_t ret = esp_ble_gap_set_device_name(config_name);
    if (ret) {
        ESP_LOGE(TAG, "Failed to set device name: %s", esp_err_to_name(ret));
        return ret;
    }
    
    // Set flag
    in_config_mode = true;
    
    // Restart advertising to update name
    esp_ble_adv_params_t adv_params = {
        .adv_int_min = 0x20,
        .adv_int_max = 0x40,
        .adv_type = ADV_TYPE_IND,
        .own_addr_type = BLE_ADDR_TYPE_PUBLIC,
        .channel_map = ADV_CHNL_ALL,
        .adv_filter_policy = ADV_FILTER_ALLOW_SCAN_ANY_CON_ANY,
    };
    ret = esp_ble_gap_start_advertising(&adv_params);
    if (ret) {
        ESP_LOGE(TAG, "Failed to start advertising: %s", esp_err_to_name(ret));
        return ret;
    }
    
    ESP_LOGI(TAG, "BLE configuration mode started");
    
    return ESP_OK;
}

/**
 * @brief Stop configuration mode
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t ble_manager_stop_config_mode(void)
{
    ESP_LOGI(TAG, "Stopping BLE configuration mode");
    
    if (!ble_initialized) {
        ESP_LOGE(TAG, "BLE not initialized");
        return ESP_ERR_INVALID_STATE;
    }
    
    if (!in_config_mode) {
        ESP_LOGW(TAG, "Not in configuration mode");
        return ESP_OK;
    }
    
    // Reset device name
    esp_err_t ret = esp_ble_gap_set_device_name(DEVICE_NAME);
    if (ret) {
        ESP_LOGE(TAG, "Failed to set device name: %s", esp_err_to_name(ret));
        return ret;
    }
    
    // Reset flag
    in_config_mode = false;
    system_state = NULL;
    
    // Restart advertising to update name
    esp_ble_adv_params_t adv_params = {
        .adv_int_min = 0x20,
        .adv_int_max = 0x40,
        .adv_type = ADV_TYPE_IND,
        .own_addr_type = BLE_ADDR_TYPE_PUBLIC,
        .channel_map = ADV_CHNL_ALL,
        .adv_filter_policy = ADV_FILTER_ALLOW_SCAN_ANY_CON_ANY,
    };
    ret = esp_ble_gap_start_advertising(&adv_params);
    if (ret) {
        ESP_LOGE(TAG, "Failed to start advertising: %s", esp_err_to_name(ret));
        return ret;
    }
    
    ESP_LOGI(TAG, "BLE configuration mode stopped");
    
    return ESP_OK;
}

/**
 * @brief Transmit data via BLE
 * 
 * @param data Data to transmit (JSON string)
 * @return esp_err_t ESP_OK on success
 */
esp_err_t ble_manager_transmit_data(const char *data)
{
    ESP_LOGI(TAG, "Transmitting data via BLE");
    
    if (!ble_initialized) {
        ESP_LOGE(TAG, "BLE not initialized");
        return ESP_ERR_INVALID_STATE;
    }
    
    if (data == NULL) {
        ESP_LOGE(TAG, "Invalid data");
        return ESP_ERR_INVALID_ARG;
    }
    
    // Update the data characteristic value
    esp_err_t ret = esp_ble_gatts_set_attr_value(char_data_handle, strlen(data), (uint8_t*)data);
    if (ret) {
        ESP_LOGE(TAG, "Failed to set attribute value: %s", esp_err_to_name(ret));
        return ret;
    }
    
    // Notify connected clients
    esp_gatt_rsp_t rsp;
    memset(&rsp, 0, sizeof(esp_gatt_rsp_t));
    
    // Get the current value
    esp_ble_gatts_get_attr_value(char_data_handle, &rsp.attr_value.len, &rsp.attr_value.value);
    
    // Send notification to all connected clients
    // In a real application, we would track client connections and only send to subscribed clients
    // For simplicity, we're assuming there's at most one client connection
    // TODO: Maintain a list of connected clients and their notification configurations
    
    ESP_LOGI(TAG, "BLE data transmission completed");
    
    return ESP_OK;
}

/**
 * @brief Deinitialize BLE manager
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t ble_manager_deinit(void)
{
    ESP_LOGI(TAG, "Deinitializing BLE manager");
    
    if (!ble_initialized) {
        ESP_LOGW(TAG, "BLE not initialized");
        return ESP_OK;
    }
    
    // Stop advertising
    esp_err_t ret = esp_ble_gap_stop_advertising();
    if (ret) {
        ESP_LOGE(TAG, "Failed to stop advertising: %s", esp_err_to_name(ret));
        return ret;
    }
    
    // Disable Bluedroid stack
    ret = esp_bluedroid_disable();
    if (ret) {
        ESP_LOGE(TAG, "Failed to disable Bluedroid: %s", esp_err_to_name(ret));
        return ret;
    }
    
    // Deinitialize Bluedroid stack
    ret = esp_bluedroid_deinit();
    if (ret) {
        ESP_LOGE(TAG, "Failed to deinitialize Bluedroid: %s", esp_err_to_name(ret));
        return ret;
    }
    
    // Disable BT controller
    ret = esp_bt_controller_disable();
    if (ret) {
        ESP_LOGE(TAG, "Failed to disable BT controller: %s", esp_err_to_name(ret));
        return ret;
    }
    
    // Deinitialize BT controller
    ret = esp_bt_controller_deinit();
    if (ret) {
        ESP_LOGE(TAG, "Failed to deinitialize BT controller: %s", esp_err_to_name(ret));
        return ret;
    }
    
    ble_initialized = false;
    in_config_mode = false;
    system_state = NULL;
    
    ESP_LOGI(TAG, "BLE manager deinitialized");
    
    return ESP_OK;
}

/**
 * @brief GAP event handler
 * 
 * @param event GAP event
 * @param param GAP parameters
 */
static void gap_event_handler(esp_gap_ble_cb_event_t event, esp_ble_gap_cb_param_t *param)
{
    switch (event) {
        case ESP_GAP_BLE_ADV_DATA_SET_COMPLETE_EVT:
            adv_config_done &= (~ADV_CONFIG_FLAG);
            if (adv_config_done == 0) {
                esp_ble_gap_start_advertising((esp_ble_adv_params_t*)param);
            }
            break;
            
        case ESP_GAP_BLE_SCAN_RSP_DATA_SET_COMPLETE_EVT:
            adv_config_done &= (~SCAN_RSP_CONFIG_FLAG);
            if (adv_config_done == 0) {
                esp_ble_gap_start_advertising((esp_ble_adv_params_t*)param);
            }
            break;
            
        case ESP_GAP_BLE_ADV_START_COMPLETE_EVT:
            if (param->adv_start_cmpl.status != ESP_BT_STATUS_SUCCESS) {
                ESP_LOGE(TAG, "Advertising start failed: %d", param->adv_start_cmpl.status);
            } else {
                ESP_LOGI(TAG, "Advertising started");
            }
            break;
            
        case ESP_GAP_BLE_ADV_STOP_COMPLETE_EVT:
            if (param->adv_stop_cmpl.status != ESP_BT_STATUS_SUCCESS) {
                ESP_LOGE(TAG, "Advertising stop failed: %d", param->adv_stop_cmpl.status);
            } else {
                ESP_LOGI(TAG, "Advertising stopped");
            }
            break;
            
        case ESP_GAP_BLE_UPDATE_CONN_PARAMS_EVT:
            ESP_LOGI(TAG, "Connection parameters updated");
            break;
            
        default:
            break;
    }
}

/**
 * @brief GATTS event handler
 * 
 * @param event GATTS event
 * @param gatts_if GATT interface
 * @param param GATTS parameters
 */
static void gatts_event_handler(esp_gatts_cb_event_t event, esp_gatt_if_t gatts_if, 
                              esp_ble_gatts_cb_param_t *param)
{
    // Forward the event to the profile handler
    gatts_profile_event_handler(event, gatts_if, param);
}

/**
 * @brief GATTS profile event handler
 * 
 * @param event GATTS event
 * @param gatts_if GATT interface
 * @param param GATTS parameters
 */
static void gatts_profile_event_handler(esp_gatts_cb_event_t event, esp_gatt_if_t gatts_if, 
                                      esp_ble_gatts_cb_param_t *param)
{
    switch (event) {
        case ESP_GATTS_REG_EVT:
            ESP_LOGI(TAG, "GATT application registered, status %d, app_id %d", 
                     param->reg.status, param->reg.app_id);
            
            // Create the service
            esp_ble_gatts_create_service(gatts_if, &(param->reg.app_id), SERVICE_UUID_SIZE);
            break;
            
        case ESP_GATTS_CREATE_EVT:
            ESP_LOGI(TAG, "Service created, status %d, service_handle %d", 
                     param->create.status, param->create.service_handle);
            
            // Save the service handle
            service_handle = param->create.service_handle;
            
            // Start the service
            esp_ble_gatts_start_service(service_handle);
            
            // Add data characteristic
            esp_ble_gatts_add_char(service_handle, char_data_uuid, 
                                  ESP_GATT_PERM_READ, 
                                  ESP_GATT_CHAR_PROP_BIT_READ | ESP_GATT_CHAR_PROP_BIT_NOTIFY,
                                  NULL, NULL);
            
            // Add config characteristic
            esp_ble_gatts_add_char(service_handle, char_config_uuid, 
                                  ESP_GATT_PERM_READ | ESP_GATT_PERM_WRITE,
                                  ESP_GATT_CHAR_PROP_BIT_READ | ESP_GATT_CHAR_PROP_BIT_WRITE,
                                  NULL, NULL);
            break;
            
        case ESP_GATTS_ADD_CHAR_EVT:
            ESP_LOGI(TAG, "Characteristic added, status %d, attr_handle %d", 
                     param->add_char.status, param->add_char.attr_handle);
            
            // Save the characteristic handle
            if (memcmp(param->add_char.char_uuid.uuid.uuid128, char_data_uuid, CHAR_UUID_SIZE) == 0) {
                char_data_handle = param->add_char.attr_handle;
                ESP_LOGI(TAG, "Data characteristic added");
            } else if (memcmp(param->add_char.char_uuid.uuid.uuid128, char_config_uuid, CHAR_UUID_SIZE) == 0) {
                char_config_handle = param->add_char.attr_handle;
                ESP_LOGI(TAG, "Config characteristic added");
            }
            break;
            
        case ESP_GATTS_START_EVT:
            ESP_LOGI(TAG, "Service started, status %d, service_handle %d", 
                     param->start.status, param->start.service_handle);
            break;
            
        case ESP_GATTS_CONNECT_EVT:
            ESP_LOGI(TAG, "BLE client connected, conn_id %d", param->connect.conn_id);
            break;
            
        case ESP_GATTS_DISCONNECT_EVT:
            ESP_LOGI(TAG, "BLE client disconnected, reason %d", param->disconnect.reason);
            
            // Restart advertising
            esp_ble_gap_start_advertising((esp_ble_adv_params_t*)NULL);
            break;
            
        case ESP_GATTS_WRITE_EVT:
            ESP_LOGI(TAG, "Write event, handle %d, len %d", 
                     param->write.handle, param->write.len);
            
            // Check if this is the config characteristic
            if (param->write.handle == char_config_handle) {
                // Process the configuration command
                process_config_command((char*)param->write.value, param->write.len);
            }
            break;
            
        case ESP_GATTS_READ_EVT:
            ESP_LOGI(TAG, "Read event, handle %d", param->read.handle);
            break;
            
        default:
            break;
    }
}

/**
 * @brief Process configuration command from BLE client
 * 
 * @param cmd Command string
 * @param len Command length
 */
static void process_config_command(const char *cmd, uint16_t len)
{
    if (cmd == NULL || len == 0) {
        ESP_LOGE(TAG, "Invalid command");
        return;
    }
    
    if (!in_config_mode || system_state == NULL) {
        ESP_LOGE(TAG, "Not in configuration mode");
        return;
    }
    
    // Null-terminate the command (safely)
    char cmd_buf[256];
    if (len >= sizeof(cmd_buf)) {
        len = sizeof(cmd_buf) - 1;
    }
    memcpy(cmd_buf, cmd, len);
    cmd_buf[len] = '\0';
    
    ESP_LOGI(TAG, "Processing configuration command: %s", cmd_buf);
    
    // Parse command (simple format: CMD:VALUE)
    char *sep = strchr(cmd_buf, ':');
    if (sep == NULL) {
        ESP_LOGE(TAG, "Invalid command format");
        return;
    }
    
    // Split command and value
    *sep = '\0';
    char *value = sep + 1;
    
    // Process different command types
    if (strcmp(cmd_buf, "WIFI_SSID") == 0) {
        // Set WiFi SSID
        strncpy(system_state->config.wifi_ssid, value, sizeof(system_state->config.wifi_ssid) - 1);
        ESP_LOGI(TAG, "WiFi SSID set to: %s", system_state->config.wifi_ssid);
    } else if (strcmp(cmd_buf, "WIFI_PASS") == 0) {
        // Set WiFi password
        strncpy(system_state->config.wifi_password, value, sizeof(system_state->config.wifi_password) - 1);
        ESP_LOGI(TAG, "WiFi password set");
    } else if (strcmp(cmd_buf, "SERVER_URL") == 0) {
        // Set server URL
        strncpy(system_state->config.server_url, value, sizeof(system_state->config.server_url) - 1);
        ESP_LOGI(TAG, "Server URL set to: %s", system_state->config.server_url);
    } else if (strcmp(cmd_buf, "COMM_MODE") == 0) {
        // Set communication mode
        int mode = atoi(value);
        if (mode >= 0 && mode <= 2) {
            system_state->config.comm_mode = (comm_mode_t)mode;
            ESP_LOGI(TAG, "Communication mode set to: %d", system_state->config.comm_mode);
        } else {
            ESP_LOGE(TAG, "Invalid communication mode: %d", mode);
        }
    } else if (strcmp(cmd_buf, "READING_INTERVAL") == 0) {
        // Set reading interval
        int interval = atoi(value);
        if (interval > 0) {
            system_state->config.reading_interval_sec = interval;
            ESP_LOGI(TAG, "Reading interval set to: %d seconds", system_state->config.reading_interval_sec);
        } else {
            ESP_LOGE(TAG, "Invalid reading interval: %d", interval);
        }
    } else if (strcmp(cmd_buf, "OCR_MIN_CONFIDENCE") == 0) {
        // Set minimum OCR confidence
        int confidence = atoi(value);
        if (confidence > 0 && confidence <= 100) {
            system_state->config.ocr_min_confidence = confidence;
            ESP_LOGI(TAG, "OCR minimum confidence set to: %d%%", system_state->config.ocr_min_confidence);
        } else {
            ESP_LOGE(TAG, "Invalid OCR confidence: %d", confidence);
        }
    } else if (strcmp(cmd_buf, "SAVE_CONFIG") == 0) {
        // Save configuration and signal completion
        ESP_LOGI(TAG, "Saving configuration");
        configuration_save(&system_state->config);
        
        // Signal configuration completion
        if (event_group != NULL) {
            xEventGroupSetBits(event_group, EVENT_CONFIG_DONE);
        }
    } else {
        // Unknown command
        ESP_LOGE(TAG, "Unknown command: %s", cmd_buf);
    }
}
