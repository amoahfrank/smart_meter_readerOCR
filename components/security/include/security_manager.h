/**
 * @file security_manager.h
 * @brief Security manager header for Smart Meter Reader OCR
 * 
 * This file defines the public interface for the security manager,
 * which handles encryption, secure storage, and TLS/SSL configuration.
 * 
 * @author Smart Meter Reader OCR Team
 * @date 2023
 */

#ifndef SECURITY_MANAGER_H
#define SECURITY_MANAGER_H

#include "esp_err.h"
#include "esp_tls.h"

/**
 * @brief Initialize the security manager
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t security_manager_init(void);

/**
 * @brief Set a secret in secure storage
 * 
 * @param key Secret key
 * @param value Secret value
 * @param value_len Length of secret value
 * @return esp_err_t ESP_OK on success
 */
esp_err_t security_manager_set_secret(const char *key, const char *value, size_t value_len);

/**
 * @brief Get a secret from secure storage
 * 
 * @param key Secret key
 * @param value Buffer to store secret value
 * @param max_len Maximum length of value buffer
 * @return esp_err_t ESP_OK on success
 */
esp_err_t security_manager_get_secret(const char *key, char *value, size_t max_len);

/**
 * @brief Delete a secret from secure storage
 * 
 * @param key Secret key
 * @return esp_err_t ESP_OK on success
 */
esp_err_t security_manager_delete_secret(const char *key);

/**
 * @brief Get a basic authentication header
 * 
 * @param header Buffer to store authentication header
 * @param max_len Maximum length of header buffer
 * @return esp_err_t ESP_OK on success
 */
esp_err_t security_manager_get_auth_header(char *header, size_t max_len);

/**
 * @brief Configure TLS for a connection
 * 
 * @param tls_cfg TLS configuration structure to populate
 * @return esp_err_t ESP_OK on success
 */
esp_err_t security_manager_configure_tls(esp_tls_cfg_t *tls_cfg);

/**
 * @brief Deinitialize the security manager
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t security_manager_deinit(void);

#endif /* SECURITY_MANAGER_H */
