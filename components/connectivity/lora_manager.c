/**
 * @file lora_manager.c
 * @brief LoRaWAN connectivity manager implementation
 * 
 * This file implements the LoRaWAN connectivity for the Smart Meter Reader.
 * It handles LoRa module initialization, connection to LoRaWAN networks,
 * and data transmission.
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
#include "driver/gpio.h"
#include "driver/spi_master.h"
#include "nvs_flash.h"

// Include component headers
#include "security_manager.h"

// Include local headers
#include "lora_manager.h"

static const char *TAG = "lora_manager";

// LoRa module SPI pins (adjust as needed for your hardware)
#define LORA_SPI_HOST           HSPI_HOST
#define LORA_PIN_MISO           GPIO_NUM_19
#define LORA_PIN_MOSI           GPIO_NUM_23
#define LORA_PIN_SCK            GPIO_NUM_18
#define LORA_PIN_CS             GPIO_NUM_5
#define LORA_PIN_RST            GPIO_NUM_14
#define LORA_PIN_DIO0           GPIO_NUM_26
#define LORA_PIN_DIO1           GPIO_NUM_33

// LoRaWAN parameters
#define LORAWAN_DEVICE_CLASS    CLASS_A
#define LORAWAN_REGION          LORAMAC_REGION_EU868
#define LORAWAN_APP_PORT        2
#define LORAWAN_CONFIRMED_MSG   true
#define LORAWAN_RETRY_COUNT     3

// LoRa status flags
#define LORA_STATUS_INITIALIZED BIT0
#define LORA_STATUS_JOINED      BIT1
#define LORA_STATUS_SENDING     BIT2
#define LORA_STATUS_TX_DONE     BIT3
#define LORA_STATUS_ERROR       BIT4

// NVS keys
#define NVS_LORA_NAMESPACE      "lora_manager"
#define NVS_DEVEUI_KEY          "deveui"
#define NVS_APPEUI_KEY          "appeui"
#define NVS_APPKEY_KEY          "appkey"

// LoRa state
static uint32_t lora_status = 0;
static EventGroupHandle_t lora_event_group = NULL;
static spi_device_handle_t spi_handle = NULL;
static TaskHandle_t lora_task_handle = NULL;
static bool is_initialized = false;

// LoRaWAN keys (in production, these would be stored securely)
static uint8_t dev_eui[8] = {0};
static uint8_t app_eui[8] = {0};
static uint8_t app_key[16] = {0};

// Forward declarations
static void lora_task(void *pvParameter);
static esp_err_t lora_spi_init(void);
static esp_err_t lora_module_init(void);
static esp_err_t lora_join_network(void);

/**
 * @brief Initialize the LoRaWAN manager
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t lora_manager_init(void)
{
    ESP_LOGI(TAG, "Initializing LoRaWAN manager");
    
    if (is_initialized) {
        ESP_LOGW(TAG, "LoRaWAN manager already initialized");
        return ESP_OK;
    }
    
    // Check if the LoRa module is physically present
    #ifndef CONFIG_LORA_ENABLED
    ESP_LOGW(TAG, "LoRa module not enabled in configuration");
    return ESP_ERR_NOT_SUPPORTED;
    #endif
    
    // Create event group
    lora_event_group = xEventGroupCreate();
    if (lora_event_group == NULL) {
        ESP_LOGE(TAG, "Failed to create event group");
        return ESP_ERR_NO_MEM;
    }
    
    // Initialize SPI bus for LoRa module
    esp_err_t ret = lora_spi_init();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize SPI: %s", esp_err_to_name(ret));
        vEventGroupDelete(lora_event_group);
        return ret;
    }
    
    // Load LoRaWAN keys from secure storage
    ret = security_manager_get_secret(NVS_DEVEUI_KEY, (char *)dev_eui, sizeof(dev_eui));
    if (ret != ESP_OK && ret != ESP_ERR_NOT_FOUND) {
        ESP_LOGE(TAG, "Failed to read DevEUI: %s", esp_err_to_name(ret));
        vEventGroupDelete(lora_event_group);
        return ret;
    }
    
    ret = security_manager_get_secret(NVS_APPEUI_KEY, (char *)app_eui, sizeof(app_eui));
    if (ret != ESP_OK && ret != ESP_ERR_NOT_FOUND) {
        ESP_LOGE(TAG, "Failed to read AppEUI: %s", esp_err_to_name(ret));
        vEventGroupDelete(lora_event_group);
        return ret;
    }
    
    ret = security_manager_get_secret(NVS_APPKEY_KEY, (char *)app_key, sizeof(app_key));
    if (ret != ESP_OK && ret != ESP_ERR_NOT_FOUND) {
        ESP_LOGE(TAG, "Failed to read AppKey: %s", esp_err_to_name(ret));
        vEventGroupDelete(lora_event_group);
        return ret;
    }
    
    // Create LoRa task
    BaseType_t task_created = xTaskCreate(
        lora_task,
        "lora_task",
        4096,
        NULL,
        5,
        &lora_task_handle
    );
    
    if (task_created != pdPASS) {
        ESP_LOGE(TAG, "Failed to create LoRa task");
        vEventGroupDelete(lora_event_group);
        return ESP_FAIL;
    }
    
    is_initialized = true;
    ESP_LOGI(TAG, "LoRaWAN manager initialized");
    
    return ESP_OK;
}

/**
 * @brief Set the LoRaWAN keys
 * 
 * @param dev_eui_hex DevEUI in hex format
 * @param app_eui_hex AppEUI in hex format
 * @param app_key_hex AppKey in hex format
 * @return esp_err_t ESP_OK on success
 */
esp_err_t lora_manager_set_keys(const char *dev_eui_hex, const char *app_eui_hex, const char *app_key_hex)
{
    ESP_LOGI(TAG, "Setting LoRaWAN keys");
    
    if (!is_initialized) {
        ESP_LOGE(TAG, "LoRaWAN manager not initialized");
        return ESP_ERR_INVALID_STATE;
    }
    
    if (dev_eui_hex == NULL || app_eui_hex == NULL || app_key_hex == NULL) {
        ESP_LOGE(TAG, "Invalid parameters");
        return ESP_ERR_INVALID_ARG;
    }
    
    // Convert hex strings to bytes
    // In a real implementation, proper hex conversion with validation would be used
    // For simplicity, we're just assuming the format is correct here
    
    // Store the keys in secure storage
    esp_err_t ret = security_manager_set_secret(NVS_DEVEUI_KEY, dev_eui_hex, strlen(dev_eui_hex));
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to store DevEUI: %s", esp_err_to_name(ret));
        return ret;
    }
    
    ret = security_manager_set_secret(NVS_APPEUI_KEY, app_eui_hex, strlen(app_eui_hex));
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to store AppEUI: %s", esp_err_to_name(ret));
        return ret;
    }
    
    ret = security_manager_set_secret(NVS_APPKEY_KEY, app_key_hex, strlen(app_key_hex));
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to store AppKey: %s", esp_err_to_name(ret));
        return ret;
    }
    
    ESP_LOGI(TAG, "LoRaWAN keys set successfully");
    
    return ESP_OK;
}

/**
 * @brief Transmit data over LoRaWAN
 * 
 * @param data Data to transmit
 * @return esp_err_t ESP_OK on success
 */
esp_err_t lora_manager_transmit_data(const char *data)
{
    ESP_LOGI(TAG, "Transmitting data over LoRaWAN");
    
    if (!is_initialized) {
        ESP_LOGE(TAG, "LoRaWAN manager not initialized");
        return ESP_ERR_INVALID_STATE;
    }
    
    if (data == NULL) {
        ESP_LOGE(TAG, "Invalid data");
        return ESP_ERR_INVALID_ARG;
    }
    
    // Check if LoRa is ready
    if (!(lora_status & LORA_STATUS_JOINED)) {
        ESP_LOGE(TAG, "LoRaWAN not joined to network");
        return ESP_ERR_INVALID_STATE;
    }
    
    if (lora_status & LORA_STATUS_SENDING) {
        ESP_LOGE(TAG, "LoRaWAN transmission already in progress");
        return ESP_ERR_INVALID_STATE;
    }
    
    // Clear previous status
    xEventGroupClearBits(lora_event_group, LORA_STATUS_TX_DONE | LORA_STATUS_ERROR);
    
    // Set sending flag
    xEventGroupSetBits(lora_event_group, LORA_STATUS_SENDING);
    
    // TODO: In a real implementation, the data would be queued for the LoRa task to send
    // For this example, we'll simulate a successful transmission
    
    // Simulate transmission delay
    vTaskDelay(pdMS_TO_TICKS(1000));
    
    // Set transmission done flag
    xEventGroupClearBits(lora_event_group, LORA_STATUS_SENDING);
    xEventGroupSetBits(lora_event_group, LORA_STATUS_TX_DONE);
    
    ESP_LOGI(TAG, "LoRaWAN transmission completed");
    
    return ESP_OK;
}

/**
 * @brief Deinitialize the LoRaWAN manager
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t lora_manager_deinit(void)
{
    ESP_LOGI(TAG, "Deinitializing LoRaWAN manager");
    
    if (!is_initialized) {
        ESP_LOGW(TAG, "LoRaWAN manager not initialized");
        return ESP_OK;
    }
    
    // Stop LoRa task
    if (lora_task_handle != NULL) {
        vTaskDelete(lora_task_handle);
        lora_task_handle = NULL;
    }
    
    // Release SPI device
    if (spi_handle != NULL) {
        spi_bus_remove_device(spi_handle);
        spi_handle = NULL;
    }
    
    // Delete event group
    if (lora_event_group != NULL) {
        vEventGroupDelete(lora_event_group);
        lora_event_group = NULL;
    }
    
    is_initialized = false;
    ESP_LOGI(TAG, "LoRaWAN manager deinitialized");
    
    return ESP_OK;
}

/**
 * @brief Initialize SPI bus for LoRa module
 * 
 * @return esp_err_t ESP_OK on success
 */
static esp_err_t lora_spi_init(void)
{
    ESP_LOGI(TAG, "Initializing SPI for LoRa module");
    
    // Configure SPI bus
    spi_bus_config_t bus_config = {
        .miso_io_num = LORA_PIN_MISO,
        .mosi_io_num = LORA_PIN_MOSI,
        .sclk_io_num = LORA_PIN_SCK,
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
        .max_transfer_sz = 0, // Use default
    };
    
    // Initialize SPI bus
    esp_err_t ret = spi_bus_initialize(LORA_SPI_HOST, &bus_config, 1);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize SPI bus: %s", esp_err_to_name(ret));
        return ret;
    }
    
    // Configure SPI device
    spi_device_interface_config_t dev_config = {
        .clock_speed_hz = 10 * 1000 * 1000, // 10 MHz
        .mode = 0,
        .spics_io_num = LORA_PIN_CS,
        .queue_size = 1,
        .flags = 0,
    };
    
    // Attach SPI device
    ret = spi_bus_add_device(LORA_SPI_HOST, &dev_config, &spi_handle);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to add SPI device: %s", esp_err_to_name(ret));
        spi_bus_free(LORA_SPI_HOST);
        return ret;
    }
    
    // Configure reset and interrupt pins
    gpio_config_t io_conf = {
        .pin_bit_mask = (1ULL << LORA_PIN_RST) | (1ULL << LORA_PIN_DIO0) | (1ULL << LORA_PIN_DIO1),
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    gpio_config(&io_conf);
    
    // Reset LoRa module
    gpio_set_direction(LORA_PIN_RST, GPIO_MODE_OUTPUT);
    gpio_set_level(LORA_PIN_RST, 0);
    vTaskDelay(pdMS_TO_TICKS(10));
    gpio_set_level(LORA_PIN_RST, 1);
    vTaskDelay(pdMS_TO_TICKS(10));
    
    ESP_LOGI(TAG, "SPI for LoRa module initialized");
    
    return ESP_OK;
}

/**
 * @brief Initialize LoRa module
 * 
 * @return esp_err_t ESP_OK on success
 */
static esp_err_t lora_module_init(void)
{
    ESP_LOGI(TAG, "Initializing LoRa module");
    
    // TODO: In a real implementation, communication with the LoRa module would happen here
    // For this example, we'll simulate a successful initialization
    
    // Simulate initialization delay
    vTaskDelay(pdMS_TO_TICKS(500));
    
    // Set initialization flag
    xEventGroupSetBits(lora_event_group, LORA_STATUS_INITIALIZED);
    
    ESP_LOGI(TAG, "LoRa module initialized");
    
    return ESP_OK;
}

/**
 * @brief Join LoRaWAN network
 * 
 * @return esp_err_t ESP_OK on success
 */
static esp_err_t lora_join_network(void)
{
    ESP_LOGI(TAG, "Joining LoRaWAN network");
    
    // Check if LoRa is initialized
    if (!(lora_status & LORA_STATUS_INITIALIZED)) {
        ESP_LOGE(TAG, "LoRa module not initialized");
        return ESP_ERR_INVALID_STATE;
    }
    
    // Check if LoRaWAN keys are set
    if (dev_eui[0] == 0 || app_eui[0] == 0 || app_key[0] == 0) {
        ESP_LOGE(TAG, "LoRaWAN keys not set");
        return ESP_ERR_INVALID_STATE;
    }
    
    // TODO: In a real implementation, OTAA join procedure would happen here
    // For this example, we'll simulate a successful join
    
    // Simulate join delay
    vTaskDelay(pdMS_TO_TICKS(1000));
    
    // Set joined flag
    xEventGroupSetBits(lora_event_group, LORA_STATUS_JOINED);
    
    ESP_LOGI(TAG, "Joined LoRaWAN network");
    
    return ESP_OK;
}

/**
 * @brief LoRa task to handle LoRaWAN operations
 * 
 * @param pvParameter Task parameters (unused)
 */
static void lora_task(void *pvParameter)
{
    ESP_LOGI(TAG, "LoRa task started");
    
    // Initialize LoRa module
    esp_err_t ret = lora_module_init();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize LoRa module: %s", esp_err_to_name(ret));
        xEventGroupSetBits(lora_event_group, LORA_STATUS_ERROR);
        vTaskDelete(NULL);
        return;
    }
    
    // Join network if keys are available
    if (dev_eui[0] != 0 && app_eui[0] != 0 && app_key[0] != 0) {
        ret = lora_join_network();
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "Failed to join LoRaWAN network: %s", esp_err_to_name(ret));
            xEventGroupSetBits(lora_event_group, LORA_STATUS_ERROR);
        }
    } else {
        ESP_LOGW(TAG, "LoRaWAN keys not set, skipping join");
    }
    
    // Main task loop
    while (1) {
        // Wait for events
        EventBits_t bits = xEventGroupWaitBits(
            lora_event_group,
            LORA_STATUS_SENDING,
            pdFALSE,
            pdFALSE,
            pdMS_TO_TICKS(1000)
        );
        
        // Check status
        lora_status = xEventGroupGetBits(lora_event_group);
        
        // Check for LoRaWAN related events here
        // In a real implementation, interrupts from the LoRa module would be handled
        
        // Sleep for a short time
        vTaskDelay(pdMS_TO_TICKS(100));
    }
    
    // Task should never reach here
    vTaskDelete(NULL);
}
