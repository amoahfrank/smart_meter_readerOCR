/**
 * @file state_machine.c
 * @brief Implementation of the state machine for Smart Meter Reader OCR
 * 
 * This file implements the main state machine that controls the device's
 * operational flow, transitioning between different states based on
 * events and conditions.
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
#include "esp_timer.h"
#include "esp_sleep.h"

// Include component headers
#include "camera.h"
#include "display.h"
#include "ocr.h"
#include "wifi_manager.h"
#include "ble_manager.h"
#include "lora_manager.h"
#include "power_mgmt.h"
#include "security_manager.h"
#include "ota_manager.h"
#include "configuration.h"

// Include local headers
#include "state_machine.h"

static const char *TAG = "state_machine";

// State machine task handle
static TaskHandle_t state_machine_task_handle = NULL;

// Event bit definitions
#define EVENT_CAPTURE_DONE     (1 << 0)
#define EVENT_OCR_DONE         (1 << 1)
#define EVENT_TRANSMISSION_DONE (1 << 2)
#define EVENT_ERROR            (1 << 3)
#define EVENT_CONFIG_DONE      (1 << 4)
#define EVENT_LOW_BATTERY      (1 << 5)
#define EVENT_OTA_AVAILABLE    (1 << 6)
#define EVENT_BUTTON_PRESSED   (1 << 7)

// Forward declarations
static void state_machine_task(void *pvParameters);
static void handle_state_init(system_state_t *state);
static void handle_state_capture(system_state_t *state);
static void handle_state_process(system_state_t *state);
static void handle_state_display(system_state_t *state);
static void handle_state_transmit(system_state_t *state);
static void handle_state_sleep(system_state_t *state);
static void handle_state_config(system_state_t *state);
static void handle_state_error(system_state_t *state);
static void handle_state_ota(system_state_t *state);
static void transition_state(system_state_t *state, device_state_t new_state);

/**
 * @brief Start the state machine
 * 
 * @param state Pointer to the system state structure
 * @param event_group Event group for inter-task communication
 * @return esp_err_t ESP_OK on success
 */
esp_err_t state_machine_start(system_state_t *state, EventGroupHandle_t event_group)
{
    ESP_LOGI(TAG, "Starting state machine");
    
    // Check if already running
    if (state_machine_task_handle != NULL) {
        ESP_LOGW(TAG, "State machine already running");
        return ESP_ERR_INVALID_STATE;
    }
    
    // Store event group in state
    state->event_group = event_group;
    
    // Create the state machine task
    BaseType_t ret = xTaskCreate(
        state_machine_task,
        "state_machine",
        4096,
        state,
        5,
        &state_machine_task_handle
    );
    
    if (ret != pdPASS) {
        ESP_LOGE(TAG, "Failed to create state machine task");
        return ESP_FAIL;
    }
    
    return ESP_OK;
}

/**
 * @brief State machine task function
 * 
 * @param pvParameters Task parameters (system_state_t pointer)
 */
static void state_machine_task(void *pvParameters)
{
    system_state_t *state = (system_state_t *)pvParameters;
    
    ESP_LOGI(TAG, "State machine task started");
    
    // Main state machine loop
    while (1) {
        ESP_LOGI(TAG, "Current state: %d", state->current_state);
        
        // Check battery level
        state->battery_level = power_mgmt_get_battery_level();
        if (state->battery_level < BATTERY_CRITICAL_THRESHOLD && 
            state->current_state != STATE_SLEEP && 
            state->current_state != STATE_ERROR) {
            ESP_LOGW(TAG, "Critical battery level: %d%%", state->battery_level);
            state->error_state = ERROR_LOW_BATTERY;
            transition_state(state, STATE_ERROR);
        }
        
        // Handle current state
        switch (state->current_state) {
            case STATE_INIT:
                handle_state_init(state);
                break;
            
            case STATE_CAPTURE:
                handle_state_capture(state);
                break;
                
            case STATE_PROCESS:
                handle_state_process(state);
                break;
                
            case STATE_DISPLAY:
                handle_state_display(state);
                break;
                
            case STATE_TRANSMIT:
                handle_state_transmit(state);
                break;
                
            case STATE_SLEEP:
                handle_state_sleep(state);
                break;
                
            case STATE_CONFIG:
                handle_state_config(state);
                break;
                
            case STATE_ERROR:
                handle_state_error(state);
                break;
                
            case STATE_OTA:
                handle_state_ota(state);
                break;
                
            default:
                ESP_LOGE(TAG, "Unknown state: %d", state->current_state);
                state->error_state = ERROR_UNKNOWN_STATE;
                transition_state(state, STATE_ERROR);
                break;
        }
        
        // Small delay to prevent tight loop
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}

/**
 * @brief Handle the INIT state
 * 
 * @param state Pointer to the system state structure
 */
static void handle_state_init(system_state_t *state)
{
    ESP_LOGI(TAG, "Handling INIT state");
    
    // Display boot screen
    display_clear();
    display_show_text("Smart Meter Reader", "Initializing...", NULL);
    display_update();
    
    // Check if OTA is available
    bool ota_available = ota_manager_check_update();
    if (ota_available) {
        ESP_LOGI(TAG, "OTA update available");
        transition_state(state, STATE_OTA);
        return;
    }
    
    // Transition to next state
    transition_state(state, STATE_CAPTURE);
}

/**
 * @brief Handle the CAPTURE state
 * 
 * @param state Pointer to the system state structure
 */
static void handle_state_capture(system_state_t *state)
{
    ESP_LOGI(TAG, "Handling CAPTURE state");
    
    // Update display
    display_clear();
    display_show_text("Smart Meter Reader", "Capturing image...", NULL);
    display_update();
    
    // Attempt to capture image (with retries)
    int retries = 3;
    camera_fb_t *fb = NULL;
    
    while (retries > 0) {
        // Capture frame
        fb = camera_capture();
        
        if (fb != NULL) {
            ESP_LOGI(TAG, "Image captured: %dx%d, %d bytes", 
                     fb->width, fb->height, fb->len);
            break;
        }
        
        ESP_LOGW(TAG, "Failed to capture image, retrying (%d attempts left)", retries);
        retries--;
        vTaskDelay(pdMS_TO_TICKS(100));
    }
    
    // Check if capture succeeded
    if (fb == NULL) {
        ESP_LOGE(TAG, "Failed to capture image after multiple attempts");
        state->error_state = ERROR_CAMERA_FAILURE;
        transition_state(state, STATE_ERROR);
        return;
    }
    
    // Pass the frame to OCR processing
    if (ocr_set_image(fb) != ESP_OK) {
        ESP_LOGE(TAG, "Failed to set image for OCR processing");
        camera_return_fb(fb);
        state->error_state = ERROR_OCR_FAILURE;
        transition_state(state, STATE_ERROR);
        return;
    }
    
    // Return the frame buffer (no longer needed)
    camera_return_fb(fb);
    
    // Transition to process state
    transition_state(state, STATE_PROCESS);
}

/**
 * @brief Handle the PROCESS state
 * 
 * @param state Pointer to the system state structure
 */
static void handle_state_process(system_state_t *state)
{
    ESP_LOGI(TAG, "Handling PROCESS state");
    
    // Update display
    display_clear();
    display_show_text("Smart Meter Reader", "Processing image...", NULL);
    display_update();
    
    // Process the image with OCR
    ocr_result_t result;
    esp_err_t ret = ocr_process_image(&result);
    
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "OCR processing failed");
        state->error_state = ERROR_OCR_FAILURE;
        transition_state(state, STATE_ERROR);
        return;
    }
    
    // Check confidence level
    if (result.confidence < state->config.ocr_min_confidence) {
        ESP_LOGW(TAG, "OCR confidence too low: %d%% (minimum: %d%%)", 
                 result.confidence, state->config.ocr_min_confidence);
        
        // If this is a retry, proceed anyway with best effort
        if (state->error_state == ERROR_LOW_CONFIDENCE) {
            ESP_LOGI(TAG, "Using best effort result after retry");
        } else {
            state->error_state = ERROR_LOW_CONFIDENCE;
            
            // Retry once with different camera settings
            ESP_LOGI(TAG, "Retrying with adjusted camera settings");
            camera_adjust_for_retry();
            transition_state(state, STATE_CAPTURE);
            return;
        }
    }
    
    // Store the reading
    memcpy(&state->last_reading, &result, sizeof(ocr_result_t));
    ESP_LOGI(TAG, "OCR Result: %s (Confidence: %d%%)", 
             state->last_reading.text, state->last_reading.confidence);
    
    // Reset error state if we had a low confidence before
    if (state->error_state == ERROR_LOW_CONFIDENCE) {
        state->error_state = ERROR_NONE;
    }
    
    // Transition to display state
    transition_state(state, STATE_DISPLAY);
}

/**
 * @brief Handle the DISPLAY state
 * 
 * @param state Pointer to the system state structure
 */
static void handle_state_display(system_state_t *state)
{
    ESP_LOGI(TAG, "Handling DISPLAY state");
    
    // Format timestamp
    char timestamp[32];
    time_t now;
    time(&now);
    struct tm timeinfo;
    localtime_r(&now, &timeinfo);
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", &timeinfo);
    
    // Format battery level
    char battery[16];
    snprintf(battery, sizeof(battery), "Batt: %d%%", state->battery_level);
    
    // Update display with reading
    display_clear();
    display_show_reading(state->last_reading.text, timestamp, battery);
    display_update();
    
    // Transition to transmit state
    transition_state(state, STATE_TRANSMIT);
}

/**
 * @brief Handle the TRANSMIT state
 * 
 * @param state Pointer to the system state structure
 */
static void handle_state_transmit(system_state_t *state)
{
    ESP_LOGI(TAG, "Handling TRANSMIT state");
    
    // Display status
    display_show_status("Transmitting...");
    display_partial_update();
    
    // Prepare data for transmission
    char data_json[256];
    snprintf(data_json, sizeof(data_json),
             "{\"reading\":\"%s\",\"confidence\":%d,\"battery\":%d,\"timestamp\":%lld}",
             state->last_reading.text,
             state->last_reading.confidence,
             state->battery_level,
             (long long)time(NULL));
    
    // Choose transmission method based on configuration
    bool transmission_success = false;
    
    switch (state->config.comm_mode) {
        case COMM_MODE_WIFI:
            // Connect to WiFi if not connected
            if (wifi_manager_connect() == ESP_OK) {
                // Transmit data
                if (wifi_manager_transmit_data(data_json) == ESP_OK) {
                    transmission_success = true;
                } else {
                    ESP_LOGW(TAG, "Failed to transmit data over WiFi");
                }
                
                // Check for OTA update
                if (ota_manager_check_update()) {
                    ESP_LOGI(TAG, "OTA update available");
                    // We'll transition to OTA state after handling current state
                    xEventGroupSetBits(state->event_group, EVENT_OTA_AVAILABLE);
                }
                
                // Disconnect WiFi to save power
                wifi_manager_disconnect();
            } else {
                ESP_LOGW(TAG, "Failed to connect to WiFi");
            }
            break;
            
        case COMM_MODE_BLE:
            if (ble_manager_transmit_data(data_json) == ESP_OK) {
                transmission_success = true;
            } else {
                ESP_LOGW(TAG, "Failed to transmit data over BLE");
            }
            break;
            
        case COMM_MODE_LORA:
            if (lora_manager_transmit_data(data_json) == ESP_OK) {
                transmission_success = true;
            } else {
                ESP_LOGW(TAG, "Failed to transmit data over LoRa");
            }
            break;
            
        default:
            ESP_LOGW(TAG, "Unknown communication mode: %d", state->config.comm_mode);
            break;
    }
    
    // Update display with transmission result
    if (transmission_success) {
        display_show_status("Transmission OK");
        ESP_LOGI(TAG, "Data transmitted successfully");
    } else {
        display_show_status("Transmission failed");
        ESP_LOGW(TAG, "Failed to transmit data");
        
        // Don't set error state, just log the failure
        // We'll still go to sleep and try again next cycle
    }
    display_partial_update();
    
    // Check for OTA event
    EventBits_t bits = xEventGroupGetBits(state->event_group);
    if (bits & EVENT_OTA_AVAILABLE) {
        transition_state(state, STATE_OTA);
    } else {
        // Otherwise, go to sleep
        transition_state(state, STATE_SLEEP);
    }
}

/**
 * @brief Handle the SLEEP state
 * 
 * @param state Pointer to the system state structure
 */
static void handle_state_sleep(system_state_t *state)
{
    ESP_LOGI(TAG, "Handling SLEEP state");
    
    // Calculate time until next reading
    int64_t sleep_time_us = state->config.reading_interval_sec * 1000000LL;
    
    // Update display before sleep
    display_show_status("Sleeping...");
    char next_wake[64];
    snprintf(next_wake, sizeof(next_wake), "Next reading in %d s", 
             state->config.reading_interval_sec);
    display_show_text(NULL, next_wake, NULL);
    display_update();
    
    // Wait for display to finish updating (e-paper displays take time)
    vTaskDelay(pdMS_TO_TICKS(100));
    
    // Prepare for deep sleep
    ESP_LOGI(TAG, "Entering deep sleep for %lld seconds", 
             sleep_time_us / 1000000);
    
    // Configure wakeup sources
    esp_sleep_enable_timer_wakeup(sleep_time_us);
    
    // Power down components
    camera_deinit();
    display_deinit();
    
    // Enter deep sleep
    esp_deep_sleep_start();
    
    // Execution will never reach here
}

/**
 * @brief Handle the CONFIG state
 * 
 * @param state Pointer to the system state structure
 */
static void handle_state_config(system_state_t *state)
{
    ESP_LOGI(TAG, "Handling CONFIG state");
    
    // Update display
    display_clear();
    display_show_text("Configuration Mode", "Connect via BLE", "SmartMeterReader");
    display_update();
    
    // Enable BLE for configuration
    ble_manager_start_config_mode(state);
    
    // Main configuration loop
    bool config_done = false;
    uint32_t timeout_counter = 0;
    const uint32_t CONFIG_TIMEOUT_SEC = 300; // 5 minutes timeout
    
    while (!config_done) {
        // Check for events
        EventBits_t bits = xEventGroupWaitBits(
            state->event_group,
            EVENT_CONFIG_DONE | EVENT_BUTTON_PRESSED,
            pdTRUE,  // Clear on exit
            pdFALSE, // Don't wait for all bits
            pdMS_TO_TICKS(1000) // 1 second timeout
        );
        
        if (bits & EVENT_CONFIG_DONE) {
            ESP_LOGI(TAG, "Configuration completed");
            config_done = true;
        } else if (bits & EVENT_BUTTON_PRESSED) {
            ESP_LOGI(TAG, "Button pressed, exiting configuration mode");
            config_done = true;
        } else {
            // Timeout, increment counter
            timeout_counter++;
            if (timeout_counter % 30 == 0) {
                // Every 30 seconds, update display (e-paper doesn't need frequent updates)
                display_show_text("Configuration Mode", 
                                 "Connect via BLE", 
                                 "Timeout in %d s", 
                                 CONFIG_TIMEOUT_SEC - timeout_counter);
                display_update();
            }
            
            // Check for timeout
            if (timeout_counter >= CONFIG_TIMEOUT_SEC) {
                ESP_LOGW(TAG, "Configuration mode timeout");
                config_done = true;
            }
        }
    }
    
    // Stop BLE configuration mode
    ble_manager_stop_config_mode();
    
    // Save configuration
    configuration_save(&state->config);
    
    // Display confirmation
    display_clear();
    display_show_text("Configuration", "Completed", "Restarting...");
    display_update();
    
    // Give time to read the display
    vTaskDelay(pdMS_TO_TICKS(2000));
    
    // Restart the device to apply new configuration
    esp_restart();
}

/**
 * @brief Handle the ERROR state
 * 
 * @param state Pointer to the system state structure
 */
static void handle_state_error(system_state_t *state)
{
    ESP_LOGI(TAG, "Handling ERROR state: %d", state->error_state);
    
    // Update display with error information
    display_clear();
    
    char *error_title = "Error";
    char *error_message = "Unknown error";
    char *error_action = "Please restart device";
    
    switch (state->error_state) {
        case ERROR_CAMERA_FAILURE:
            error_message = "Camera failure";
            error_action = "Check camera connection";
            break;
            
        case ERROR_OCR_FAILURE:
            error_message = "OCR processing failed";
            error_action = "Retrying in 60 seconds";
            break;
            
        case ERROR_LOW_CONFIDENCE:
            error_message = "Reading confidence low";
            error_action = "Retrying with new settings";
            break;
            
        case ERROR_COMMUNICATION:
            error_message = "Communication failure";
            error_action = "Check network settings";
            break;
            
        case ERROR_LOW_BATTERY:
            error_message = "Battery critically low";
            error_action = "Charge battery";
            break;
            
        case ERROR_UNKNOWN_STATE:
            error_message = "System in unknown state";
            error_action = "Restarting system";
            break;
            
        case ERROR_OTA_FAILURE:
            error_message = "OTA update failed";
            error_action = "Will retry next cycle";
            break;
            
        default:
            break;
    }
    
    display_show_text(error_title, error_message, error_action);
    display_update();
    
    // Handle different error types
    switch (state->error_state) {
        case ERROR_CAMERA_FAILURE:
            // Critical error, need restart
            vTaskDelay(pdMS_TO_TICKS(5000)); // Show error for 5 seconds
            esp_restart();
            break;
            
        case ERROR_OCR_FAILURE:
            // Wait and retry
            vTaskDelay(pdMS_TO_TICKS(10000)); // Show error for 10 seconds
            transition_state(state, STATE_CAPTURE);
            break;
            
        case ERROR_LOW_CONFIDENCE:
            // Will be handled in process state
            transition_state(state, STATE_CAPTURE);
            break;
            
        case ERROR_COMMUNICATION:
            // Non-critical, continue to sleep
            vTaskDelay(pdMS_TO_TICKS(5000)); // Show error for 5 seconds
            transition_state(state, STATE_SLEEP);
            break;
            
        case ERROR_LOW_BATTERY:
            // Critical error, go to sleep to conserve power
            vTaskDelay(pdMS_TO_TICKS(5000)); // Show error for 5 seconds
            
            // Set a longer sleep interval to allow for charging
            int64_t emergency_sleep_time = 3600 * 1000000LL; // 1 hour
            esp_sleep_enable_timer_wakeup(emergency_sleep_time);
            
            // Power down all components
            camera_deinit();
            display_deinit();
            wifi_manager_disconnect();
            ble_manager_deinit();
            
            // Enter deep sleep
            esp_deep_sleep_start();
            break;
            
        case ERROR_UNKNOWN_STATE:
            // Critical error, need restart
            vTaskDelay(pdMS_TO_TICKS(5000)); // Show error for 5 seconds
            esp_restart();
            break;
            
        case ERROR_OTA_FAILURE:
            // Non-critical, continue to sleep
            vTaskDelay(pdMS_TO_TICKS(5000)); // Show error for 5 seconds
            transition_state(state, STATE_SLEEP);
            break;
            
        default:
            // Unknown error, restart to be safe
            vTaskDelay(pdMS_TO_TICKS(5000)); // Show error for 5 seconds
            esp_restart();
            break;
    }
}

/**
 * @brief Handle the OTA state
 * 
 * @param state Pointer to the system state structure
 */
static void handle_state_ota(system_state_t *state)
{
    ESP_LOGI(TAG, "Handling OTA state");
    
    // Update display
    display_clear();
    display_show_text("OTA Update", "Downloading...", "Please wait");
    display_update();
    
    // Ensure WiFi is connected
    if (wifi_manager_connect() != ESP_OK) {
        ESP_LOGE(TAG, "Failed to connect to WiFi for OTA");
        state->error_state = ERROR_OTA_FAILURE;
        transition_state(state, STATE_ERROR);
        return;
    }
    
    // Perform OTA update
    esp_err_t ret = ota_manager_perform_update();
    
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "OTA update failed");
        wifi_manager_disconnect();
        state->error_state = ERROR_OTA_FAILURE;
        transition_state(state, STATE_ERROR);
        return;
    }
    
    // Update successful, restart the device
    display_clear();
    display_show_text("OTA Update", "Success", "Restarting...");
    display_update();
    
    // Wait for display to update
    vTaskDelay(pdMS_TO_TICKS(2000));
    
    // Disconnect WiFi
    wifi_manager_disconnect();
    
    // Restart to apply the update
    esp_restart();
}

/**
 * @brief Transition to a new state
 * 
 * @param state Pointer to the system state structure
 * @param new_state The new state to transition to
 */
static void transition_state(system_state_t *state, device_state_t new_state)
{
    ESP_LOGI(TAG, "State transition: %d -> %d", state->current_state, new_state);
    state->current_state = new_state;
}
