/**
 * @file state_machine.h
 * @brief State machine header for Smart Meter Reader OCR
 * 
 * This file defines the public interface for the state machine,
 * which controls the operational flow of the device.
 * 
 * @author Frank Amoah A.K.A SiRWaTT Smart Meter Reader OCR Team Lead
 * @date 2023
 */

#ifndef STATE_MACHINE_H
#define STATE_MACHINE_H

#include "esp_err.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"

#include "app_main.h"  // For system_state_t

// Event bits
#define EVENT_CONFIG_DONE      (1 << 4)
#define EVENT_BUTTON_PRESSED   (1 << 7)

/**
 * @brief Start the state machine
 * 
 * @param state Pointer to the system state structure
 * @param event_group Event group for inter-task communication
 * @return esp_err_t ESP_OK on success
 */
esp_err_t state_machine_start(system_state_t *state, EventGroupHandle_t event_group);

#endif /* STATE_MACHINE_H */
