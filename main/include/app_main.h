/**
 * @file app_main.h
 * @brief Main application header for Smart Meter Reader OCR
 * 
 * This file contains the definitions and declarations for the main
 * application, including system state and event definitions.
 * 
 * @author Frank Amoah A.K.A SiRWaTT Smart Meter Reader OCR Team Lead
 * @date 2023
 */

#ifndef APP_MAIN_H
#define APP_MAIN_H

#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "esp_err.h"

// Forward declarations
#include "configuration.h"  // For device_config_t
#include "ocr.h"           // For ocr_result_t

/**
 * @brief Device state enumeration
 */
typedef enum {
    STATE_INIT = 0,        /*!< Initialization state */
    STATE_CAPTURE,         /*!< Camera capture state */
    STATE_PROCESS,         /*!< OCR processing state */
    STATE_DISPLAY,         /*!< Display update state */
    STATE_TRANSMIT,        /*!< Data transmission state */
    STATE_SLEEP,           /*!< Sleep state */
    STATE_CONFIG,          /*!< Configuration state */
    STATE_ERROR,           /*!< Error state */
    STATE_OTA,             /*!< OTA update state */
} device_state_t;

/**
 * @brief Error state enumeration
 */
typedef enum {
    ERROR_NONE = 0,               /*!< No error */
    ERROR_CAMERA_FAILURE,         /*!< Camera failure */
    ERROR_OCR_FAILURE,            /*!< OCR processing failure */
    ERROR_LOW_CONFIDENCE,         /*!< Low OCR confidence */
    ERROR_COMMUNICATION,          /*!< Communication failure */
    ERROR_LOW_BATTERY,            /*!< Low battery */
    ERROR_UNKNOWN_STATE,          /*!< Unknown state */
    ERROR_OTA_FAILURE,            /*!< OTA update failure */
} error_state_t;

/**
 * @brief System state structure
 */
typedef struct {
    device_state_t current_state;      /*!< Current device state */
    int battery_level;                 /*!< Battery level (0-100) */
    ocr_result_t last_reading;         /*!< Last OCR reading */
    error_state_t error_state;         /*!< Current error state */
    device_config_t config;            /*!< Device configuration */
    EventGroupHandle_t event_group;    /*!< Event group for communication */
} system_state_t;

/**
 * @brief Application entry point
 */
void app_main(void);

#endif /* APP_MAIN_H */
