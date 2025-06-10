/**
 * @file wifi_manager.c
 * @brief WiFi connectivity manager implementation
 * 
 * This file implements the WiFi connectivity manager for the Smart Meter Reader.
 * It handles WiFi initialization, connection, data transmission, and power management.
 * 
 * @author Frank Amoah A.K.A SiRWaTT Smart Meter Reader OCR Team Lead
 * @date 2023
 */

#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "esp_system.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_http_client.h"
#include "nvs_flash.h"
#include "lwip/err.h"
#include "lwip/sys.h"

// Include component headers
#include "security_manager.h"

// Include local headers
#include "wifi_manager.h"

static const char *TAG = "wifi_manager";

// WiFi connection status event bits
#define WIFI_CONNECTED_BIT      BIT0
#define WIFI_FAIL_BIT           BIT1
#define WIFI_MAXIMUM_RETRY      5

// WiFi event group
static EventGroupHandle_t wifi_event_group;
static int s_retry_num = 0;
static bool wifi_initialized = false;
static bool wifi_connected = false;

// Server endpoint configuration
static char server_url[256] = CONFIG_DEFAULT_SERVER_URL;

// Forward declarations
static void event_handler(void* arg, esp_event_base_t event_base,
                         int32_t event_id, void* event_data);
static esp_err_t http_event_handler(esp_http_client_event_t *evt);

/**
 * @brief Initialize the WiFi subsystem
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t wifi_manager_init(void)
{
    ESP_LOGI(TAG, "Initializing WiFi manager");
    
    if (wifi_initialized) {
        ESP_LOGW(TAG, "WiFi already initialized");
        return ESP_OK;
    }
    
    // Create WiFi event group
    wifi_event_group = xEventGroupCreate();
    
    // Initialize networking stack
    ESP_ERROR_CHECK(esp_netif_init());
    
    // Initialize event loop
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    
    // Create default WiFi station
    esp_netif_create_default_wifi_sta();
    
    // Initialize WiFi with default configuration
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    
    // Register event handlers
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &event_handler, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &event_handler, NULL));
    
    // Configure WiFi in station mode
    wifi_config_t wifi_config = {
        .sta = {
            .ssid = "",
            .password = "",
            .threshold.authmode = WIFI_AUTH_WPA2_PSK,
            .pmf_cfg = {
                .capable = true,
                .required = false
            },
        },
    };
    
    // Set WiFi mode to station
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    
    // Set WiFi configuration
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    
    // Set WiFi power save mode
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_MAX_MODEM));
    
    wifi_initialized = true;
    ESP_LOGI(TAG, "WiFi manager initialized");
    
    return ESP_OK;
}

/**
 * @brief Connect to WiFi network
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t wifi_manager_connect(void)
{
    ESP_LOGI(TAG, "Connecting to WiFi");
    
    if (!wifi_initialized) {
        ESP_LOGE(TAG, "WiFi not initialized");
        return ESP_ERR_INVALID_STATE;
    }
    
    // If already connected, return success
    if (wifi_connected) {
        ESP_LOGI(TAG, "WiFi already connected");
        return ESP_OK;
    }
    
    // Reset retry counter
    s_retry_num = 0;
    
    // Clear event bits
    xEventGroupClearBits(wifi_event_group, WIFI_CONNECTED_BIT | WIFI_FAIL_BIT);
    
    // Load WiFi credentials from NVS
    wifi_config_t wifi_config;
    ESP_ERROR_CHECK(esp_wifi_get_config(WIFI_IF_STA, &wifi_config));
    
    // Check if we have credentials
    if (strlen((char*)wifi_config.sta.ssid) == 0) {
        ESP_LOGE(TAG, "No WiFi credentials configured");
        return ESP_ERR_INVALID_STATE;
    }
    
    ESP_LOGI(TAG, "Connecting to SSID: %s", wifi_config.sta.ssid);
    
    // Start WiFi
    ESP_ERROR_CHECK(esp_wifi_start());
    
    // Wait for connection or failure
    EventBits_t bits = xEventGroupWaitBits(wifi_event_group,
                                          WIFI_CONNECTED_BIT | WIFI_FAIL_BIT,
                                          pdFALSE,
                                          pdFALSE,
                                          pdMS_TO_TICKS(10000));
    
    // Check result
    if (bits & WIFI_CONNECTED_BIT) {
        ESP_LOGI(TAG, "Connected to AP SSID: %s", wifi_config.sta.ssid);
        wifi_connected = true;
        return ESP_OK;
    } else if (bits & WIFI_FAIL_BIT) {
        ESP_LOGE(TAG, "Failed to connect to SSID: %s", wifi_config.sta.ssid);
        return ESP_FAIL;
    } else {
        ESP_LOGE(TAG, "WiFi connection timeout");
        return ESP_ERR_TIMEOUT;
    }
}

/**
 * @brief Disconnect from WiFi network
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t wifi_manager_disconnect(void)
{
    ESP_LOGI(TAG, "Disconnecting from WiFi");
    
    if (!wifi_initialized) {
        ESP_LOGE(TAG, "WiFi not initialized");
        return ESP_ERR_INVALID_STATE;
    }
    
    if (!wifi_connected) {
        ESP_LOGW(TAG, "WiFi not connected");
        return ESP_OK;
    }
    
    // Stop WiFi
    ESP_ERROR_CHECK(esp_wifi_stop());
    
    wifi_connected = false;
    ESP_LOGI(TAG, "WiFi disconnected");
    
    return ESP_OK;
}

/**
 * @brief Set WiFi credentials
 * 
 * @param ssid WiFi SSID
 * @param password WiFi password
 * @return esp_err_t ESP_OK on success
 */
esp_err_t wifi_manager_set_credentials(const char *ssid, const char *password)
{
    ESP_LOGI(TAG, "Setting WiFi credentials: SSID=%s", ssid);
    
    if (!wifi_initialized) {
        ESP_LOGE(TAG, "WiFi not initialized");
        return ESP_ERR_INVALID_STATE;
    }
    
    if (ssid == NULL || password == NULL) {
        ESP_LOGE(TAG, "Invalid WiFi credentials");
        return ESP_ERR_INVALID_ARG;
    }
    
    // Prepare WiFi configuration
    wifi_config_t wifi_config = {
        .sta = {
            .threshold.authmode = WIFI_AUTH_WPA2_PSK,
            .pmf_cfg = {
                .capable = true,
                .required = false
            },
        },
    };
    
    // Copy SSID and password
    strncpy((char*)wifi_config.sta.ssid, ssid, sizeof(wifi_config.sta.ssid) - 1);
    strncpy((char*)wifi_config.sta.password, password, sizeof(wifi_config.sta.password) - 1);
    
    // Set WiFi configuration
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    
    ESP_LOGI(TAG, "WiFi credentials set successfully");
    
    return ESP_OK;
}

/**
 * @brief Set server URL
 * 
 * @param url Server URL
 * @return esp_err_t ESP_OK on success
 */
esp_err_t wifi_manager_set_server_url(const char *url)
{
    ESP_LOGI(TAG, "Setting server URL: %s", url);
    
    if (url == NULL) {
        ESP_LOGE(TAG, "Invalid server URL");
        return ESP_ERR_INVALID_ARG;
    }
    
    // Copy server URL
    strncpy(server_url, url, sizeof(server_url) - 1);
    
    return ESP_OK;
}

/**
 * @brief Transmit data to server
 * 
 * @param data Data to transmit (JSON string)
 * @return esp_err_t ESP_OK on success
 */
esp_err_t wifi_manager_transmit_data(const char *data)
{
    ESP_LOGI(TAG, "Transmitting data to server");
    
    if (!wifi_connected) {
        ESP_LOGE(TAG, "WiFi not connected");
        return ESP_ERR_INVALID_STATE;
    }
    
    if (data == NULL) {
        ESP_LOGE(TAG, "Invalid data");
        return ESP_ERR_INVALID_ARG;
    }
    
    // Configure HTTP client
    esp_http_client_config_t config = {
        .url = server_url,
        .method = HTTP_METHOD_POST,
        .event_handler = http_event_handler,
        .timeout_ms = 10000,
        .disable_auto_redirect = true,
    };
    
    // Create HTTP client
    esp_http_client_handle_t client = esp_http_client_init(&config);
    
    // Set headers
    esp_http_client_set_header(client, "Content-Type", "application/json");
    
    // Add authentication (if configured)
    char auth_header[128];
    if (security_manager_get_auth_header(auth_header, sizeof(auth_header)) == ESP_OK) {
        esp_http_client_set_header(client, "Authorization", auth_header);
    }
    
    // Set post data
    esp_http_client_set_post_field(client, data, strlen(data));
    
    // Perform HTTP request
    esp_err_t err = esp_http_client_perform(client);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "HTTP POST request failed: %s", esp_err_to_name(err));
        esp_http_client_cleanup(client);
        return err;
    }
    
    // Check status code
    int status_code = esp_http_client_get_status_code(client);
    ESP_LOGI(TAG, "HTTP POST status = %d", status_code);
    
    // Clean up
    esp_http_client_cleanup(client);
    
    // Return error if status code is not 2xx
    if (status_code < 200 || status_code >= 300) {
        ESP_LOGE(TAG, "HTTP POST request returned non-2xx status code: %d", status_code);
        return ESP_FAIL;
    }
    
    ESP_LOGI(TAG, "Data transmitted successfully");
    
    return ESP_OK;
}

/**
 * @brief Deinitialize WiFi manager
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t wifi_manager_deinit(void)
{
    ESP_LOGI(TAG, "Deinitializing WiFi manager");
    
    if (!wifi_initialized) {
        ESP_LOGW(TAG, "WiFi not initialized");
        return ESP_OK;
    }
    
    // Disconnect if connected
    if (wifi_connected) {
        wifi_manager_disconnect();
    }
    
    // Unregister event handlers
    ESP_ERROR_CHECK(esp_event_handler_unregister(IP_EVENT, IP_EVENT_STA_GOT_IP, &event_handler));
    ESP_ERROR_CHECK(esp_event_handler_unregister(WIFI_EVENT, ESP_EVENT_ANY_ID, &event_handler));
    
    // Deinitialize WiFi
    ESP_ERROR_CHECK(esp_wifi_deinit());
    
    // Delete event group
    vEventGroupDelete(wifi_event_group);
    
    wifi_initialized = false;
    ESP_LOGI(TAG, "WiFi manager deinitialized");
    
    return ESP_OK;
}

/**
 * @brief WiFi event handler
 * 
 * @param arg Event argument
 * @param event_base Event base
 * @param event_id Event ID
 * @param event_data Event data
 */
static void event_handler(void* arg, esp_event_base_t event_base,
                         int32_t event_id, void* event_data)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        ESP_LOGI(TAG, "WiFi station started, connecting to AP");
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        wifi_event_sta_disconnected_t* event = (wifi_event_sta_disconnected_t*) event_data;
        
        if (s_retry_num < WIFI_MAXIMUM_RETRY) {
            ESP_LOGW(TAG, "WiFi disconnected (reason %d), retrying (%d/%d)",
                     event->reason, s_retry_num + 1, WIFI_MAXIMUM_RETRY);
            esp_wifi_connect();
            s_retry_num++;
        } else {
            ESP_LOGE(TAG, "WiFi connection failed after maximum retries");
            xEventGroupSetBits(wifi_event_group, WIFI_FAIL_BIT);
        }
        
        wifi_connected = false;
        ESP_LOGI(TAG, "WiFi disconnected");
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t* event = (ip_event_got_ip_t*) event_data;
        ESP_LOGI(TAG, "WiFi connected, got IP: " IPSTR, IP2STR(&event->ip_info.ip));
        s_retry_num = 0;
        xEventGroupSetBits(wifi_event_group, WIFI_CONNECTED_BIT);
        wifi_connected = true;
    }
}

/**
 * @brief HTTP client event handler
 * 
 * @param evt HTTP client event
 * @return esp_err_t ESP_OK on success
 */
static esp_err_t http_event_handler(esp_http_client_event_t *evt)
{
    switch(evt->event_id) {
        case HTTP_EVENT_ERROR:
            ESP_LOGW(TAG, "HTTP client error");
            break;
        case HTTP_EVENT_ON_CONNECTED:
            ESP_LOGI(TAG, "HTTP client connected");
            break;
        case HTTP_EVENT_HEADER_SENT:
            ESP_LOGI(TAG, "HTTP headers sent");
            break;
        case HTTP_EVENT_ON_HEADER:
            ESP_LOGD(TAG, "HTTP header received: %s: %s", evt->header_key, evt->header_value);
            break;
        case HTTP_EVENT_ON_DATA:
            ESP_LOGI(TAG, "HTTP data received (%d bytes)", evt->data_len);
            // Process response data if needed
            break;
        case HTTP_EVENT_ON_FINISH:
            ESP_LOGI(TAG, "HTTP request finished");
            break;
        case HTTP_EVENT_DISCONNECTED:
            ESP_LOGI(TAG, "HTTP client disconnected");
            break;
    }
    return ESP_OK;
}
