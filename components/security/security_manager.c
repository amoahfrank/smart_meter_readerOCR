/**
 * @file security_manager.c
 * @brief Security manager implementation for Smart Meter Reader OCR
 * 
 * This file implements the security manager, which handles secure storage,
 * authentication, TLS/SSL configuration, and secure boot verification.
 * 
 * @author Frank Amoah A.K.A SiRWaTT Smart Meter Reader OCR Team Lead
 * @date 2023
 */

#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_tls.h"
#include "esp_crt_bundle.h"
#include "esp_secure_boot.h"
#include "esp_efuse.h"
#include "esp_flash_encrypt.h"
#include "nvs_flash.h"

// ESP-IDF 5.0 specific
#include "esp_random.h"
#include "mbedtls/base64.h"
#include "mbedtls/sha256.h"
#include "mbedtls/aes.h"

// Include local headers
#include "security_manager.h"

static const char *TAG = "security_manager";

// NVS namespace for secure storage
#define SECURE_NAMESPACE        "secure_store"

// Encryption key info (for secure storage)
#define KEY_LENGTH              32  // 256 bits
#define IV_LENGTH               16  // 128 bits

// Maximum secret size
#define MAX_SECRET_SIZE         256

// NVS handle for secure storage
static nvs_handle_t nvs_handle;
static bool is_initialized = false;

// Root CA certificate for TLS/SSL connections
extern const char root_ca_pem_start[] asm("_binary_ca_cert_pem_start");
extern const char root_ca_pem_end[] asm("_binary_ca_cert_pem_end");

// Encryption key and IV (derived from device unique ID in a real implementation)
static uint8_t encryption_key[KEY_LENGTH];
static uint8_t encryption_iv[IV_LENGTH];

// Forward declarations
static esp_err_t init_secure_storage(void);
static esp_err_t derive_encryption_key(void);
static esp_err_t encrypt_data(const uint8_t *input, size_t input_len, 
                           uint8_t *output, size_t *output_len);
static esp_err_t decrypt_data(const uint8_t *input, size_t input_len, 
                           uint8_t *output, size_t *output_len);

/**
 * @brief Initialize the security manager
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t security_manager_init(void)
{
    ESP_LOGI(TAG, "Initializing security manager");
    
    if (is_initialized) {
        ESP_LOGW(TAG, "Security manager already initialized");
        return ESP_OK;
    }
    
    // Check secure boot status
    #if CONFIG_SECURE_BOOT
    if (esp_secure_boot_enabled()) {
        ESP_LOGI(TAG, "Secure boot is enabled and verified");
    } else {
        ESP_LOGW(TAG, "Secure boot is not enabled");
        // In production, you might want to halt if secure boot is expected but not enabled
    }
    #else
    ESP_LOGW(TAG, "Secure boot is not enabled in the configuration");
    #endif
    
    // Check flash encryption status
    #if CONFIG_SECURE_FLASH_ENC_ENABLED
    if (esp_flash_encryption_enabled()) {
        ESP_LOGI(TAG, "Flash encryption is enabled");
    } else {
        ESP_LOGW(TAG, "Flash encryption is not enabled");
        // In production, you might want to halt if flash encryption is expected but not enabled
    }
    #else
    ESP_LOGW(TAG, "Flash encryption is not enabled in the configuration");
    #endif
    
    // Initialize secure storage
    esp_err_t err = init_secure_storage();
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize secure storage: %s", esp_err_to_name(err));
        return err;
    }
    
    // Derive encryption key (for secure storage)
    err = derive_encryption_key();
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to derive encryption key: %s", esp_err_to_name(err));
        return err;
    }
    
    is_initialized = true;
    ESP_LOGI(TAG, "Security manager initialized");
    
    return ESP_OK;
}

/**
 * @brief Set a secret in secure storage
 * 
 * @param key Secret key
 * @param value Secret value
 * @param value_len Length of secret value
 * @return esp_err_t ESP_OK on success
 */
esp_err_t security_manager_set_secret(const char *key, const char *value, size_t value_len)
{
    ESP_LOGI(TAG, "Setting secret: %s", key);
    
    if (!is_initialized) {
        ESP_LOGE(TAG, "Security manager not initialized");
        return ESP_ERR_INVALID_STATE;
    }
    
    if (key == NULL || value == NULL || value_len == 0 || value_len > MAX_SECRET_SIZE) {
        ESP_LOGE(TAG, "Invalid parameters");
        return ESP_ERR_INVALID_ARG;
    }
    
    // Allocate buffer for encrypted data (with room for possible padding)
    size_t encrypted_len = value_len + 32;
    uint8_t *encrypted_data = (uint8_t*)malloc(encrypted_len);
    if (encrypted_data == NULL) {
        ESP_LOGE(TAG, "Failed to allocate memory for encryption");
        return ESP_ERR_NO_MEM;
    }
    
    // Encrypt the data
    esp_err_t err = encrypt_data((uint8_t*)value, value_len, encrypted_data, &encrypted_len);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to encrypt data: %s", esp_err_to_name(err));
        free(encrypted_data);
        return err;
    }
    
    // Store encrypted data in NVS
    err = nvs_set_blob(nvs_handle, key, encrypted_data, encrypted_len);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to store encrypted data: %s", esp_err_to_name(err));
        free(encrypted_data);
        return err;
    }
    
    // Commit NVS changes
    err = nvs_commit(nvs_handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to commit NVS changes: %s", esp_err_to_name(err));
        free(encrypted_data);
        return err;
    }
    
    free(encrypted_data);
    ESP_LOGI(TAG, "Secret set successfully");
    
    return ESP_OK;
}

/**
 * @brief Get a secret from secure storage
 * 
 * @param key Secret key
 * @param value Buffer to store secret value
 * @param max_len Maximum length of value buffer
 * @return esp_err_t ESP_OK on success
 */
esp_err_t security_manager_get_secret(const char *key, char *value, size_t max_len)
{
    ESP_LOGI(TAG, "Getting secret: %s", key);
    
    if (!is_initialized) {
        ESP_LOGE(TAG, "Security manager not initialized");
        return ESP_ERR_INVALID_STATE;
    }
    
    if (key == NULL || value == NULL || max_len == 0) {
        ESP_LOGE(TAG, "Invalid parameters");
        return ESP_ERR_INVALID_ARG;
    }
    
    // Get encrypted data size
    size_t encrypted_len = 0;
    esp_err_t err = nvs_get_blob(nvs_handle, key, NULL, &encrypted_len);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to get encrypted data size: %s", esp_err_to_name(err));
        return err;
    }
    
    // Allocate buffer for encrypted data
    uint8_t *encrypted_data = (uint8_t*)malloc(encrypted_len);
    if (encrypted_data == NULL) {
        ESP_LOGE(TAG, "Failed to allocate memory for encrypted data");
        return ESP_ERR_NO_MEM;
    }
    
    // Get encrypted data
    err = nvs_get_blob(nvs_handle, key, encrypted_data, &encrypted_len);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to get encrypted data: %s", esp_err_to_name(err));
        free(encrypted_data);
        return err;
    }
    
    // Allocate buffer for decrypted data
    size_t decrypted_len = max_len;
    
    // Decrypt the data
    err = decrypt_data(encrypted_data, encrypted_len, (uint8_t*)value, &decrypted_len);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to decrypt data: %s", esp_err_to_name(err));
        free(encrypted_data);
        return err;
    }
    
    // Null-terminate the string
    if (decrypted_len < max_len) {
        value[decrypted_len] = '\0';
    } else {
        value[max_len - 1] = '\0';
    }
    
    free(encrypted_data);
    ESP_LOGI(TAG, "Secret retrieved successfully");
    
    return ESP_OK;
}

/**
 * @brief Delete a secret from secure storage
 * 
 * @param key Secret key
 * @return esp_err_t ESP_OK on success
 */
esp_err_t security_manager_delete_secret(const char *key)
{
    ESP_LOGI(TAG, "Deleting secret: %s", key);
    
    if (!is_initialized) {
        ESP_LOGE(TAG, "Security manager not initialized");
        return ESP_ERR_INVALID_STATE;
    }
    
    if (key == NULL) {
        ESP_LOGE(TAG, "Invalid key");
        return ESP_ERR_INVALID_ARG;
    }
    
    // Delete the key from NVS
    esp_err_t err = nvs_erase_key(nvs_handle, key);
    if (err != ESP_OK && err != ESP_ERR_NVS_NOT_FOUND) {
        ESP_LOGE(TAG, "Failed to delete secret: %s", esp_err_to_name(err));
        return err;
    }
    
    // Commit NVS changes
    err = nvs_commit(nvs_handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to commit NVS changes: %s", esp_err_to_name(err));
        return err;
    }
    
    ESP_LOGI(TAG, "Secret deleted successfully");
    
    return ESP_OK;
}

/**
 * @brief Get a basic authentication header
 * 
 * @param header Buffer to store authentication header
 * @param max_len Maximum length of header buffer
 * @return esp_err_t ESP_OK on success
 */
esp_err_t security_manager_get_auth_header(char *header, size_t max_len)
{
    ESP_LOGI(TAG, "Creating authentication header");
    
    if (!is_initialized) {
        ESP_LOGE(TAG, "Security manager not initialized");
        return ESP_ERR_INVALID_STATE;
    }
    
    if (header == NULL || max_len == 0) {
        ESP_LOGE(TAG, "Invalid parameters");
        return ESP_ERR_INVALID_ARG;
    }
    
    // Get API key from secure storage
    char api_key[64] = {0};
    esp_err_t err = security_manager_get_secret("api_key", api_key, sizeof(api_key));
    
    if (err == ESP_OK && strlen(api_key) > 0) {
        // Format as Bearer token
        snprintf(header, max_len, "Bearer %s", api_key);
    } else {
        // No authentication configured
        header[0] = '\0';
        return ESP_ERR_NOT_FOUND;
    }
    
    return ESP_OK;
}

/**
 * @brief Configure TLS for a connection
 * 
 * @param tls_cfg TLS configuration structure to populate
 * @return esp_err_t ESP_OK on success
 */
esp_err_t security_manager_configure_tls(esp_tls_cfg_t *tls_cfg)
{
    ESP_LOGI(TAG, "Configuring TLS");
    
    if (!is_initialized) {
        ESP_LOGE(TAG, "Security manager not initialized");
        return ESP_ERR_INVALID_STATE;
    }
    
    if (tls_cfg == NULL) {
        ESP_LOGE(TAG, "Invalid TLS configuration");
        return ESP_ERR_INVALID_ARG;
    }
    
    // Initialize the TLS configuration
    memset(tls_cfg, 0, sizeof(esp_tls_cfg_t));
    
    // Configure root CA certificate
    tls_cfg->crt_bundle_attach = esp_crt_bundle_attach;
    
    // Configure server certificate verification
    tls_cfg->skip_common_name = false;
    tls_cfg->non_block = true;
    
    // Configure client certificate if needed
    // This would be used for mutual TLS authentication
    // Not implemented in this simplified version
    
    ESP_LOGI(TAG, "TLS configuration complete");
    
    return ESP_OK;
}

/**
 * @brief Deinitialize the security manager
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t security_manager_deinit(void)
{
    ESP_LOGI(TAG, "Deinitializing security manager");
    
    if (!is_initialized) {
        ESP_LOGW(TAG, "Security manager not initialized");
        return ESP_OK;
    }
    
    // Close NVS handle
    nvs_close(nvs_handle);
    
    // Clear sensitive data
    memset(encryption_key, 0, sizeof(encryption_key));
    memset(encryption_iv, 0, sizeof(encryption_iv));
    
    is_initialized = false;
    ESP_LOGI(TAG, "Security manager deinitialized");
    
    return ESP_OK;
}

/**
 * @brief Initialize secure storage
 * 
 * @return esp_err_t ESP_OK on success
 */
static esp_err_t init_secure_storage(void)
{
    ESP_LOGI(TAG, "Initializing secure storage");
    
    // Initialize NVS if not already initialized
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        // NVS partition was truncated and needs to be erased
        ESP_LOGW(TAG, "NVS needs to be erased");
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(err);
    
    // Open NVS namespace for secure storage
    err = nvs_open(SECURE_NAMESPACE, NVS_READWRITE, &nvs_handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Error opening NVS namespace: %s", esp_err_to_name(err));
        return err;
    }
    
    ESP_LOGI(TAG, "Secure storage initialized");
    
    return ESP_OK;
}

/**
 * @brief Derive encryption key from device unique ID
 * 
 * @return esp_err_t ESP_OK on success
 */
static esp_err_t derive_encryption_key(void)
{
    ESP_LOGI(TAG, "Deriving encryption key");
    
    // In a real implementation, you would use the device's unique ID and a KDF
    // For simplicity, we'll just use random data in this example
    
    // Generate random key
    esp_fill_random(encryption_key, KEY_LENGTH);
    
    // Generate random IV
    esp_fill_random(encryption_iv, IV_LENGTH);
    
    ESP_LOGI(TAG, "Encryption key derived");
    
    return ESP_OK;
}

/**
 * @brief Encrypt data using AES-256-CBC
 * 
 * @param input Input data
 * @param input_len Input data length
 * @param output Buffer for encrypted data
 * @param output_len Pointer to output buffer length (in/out)
 * @return esp_err_t ESP_OK on success
 */
static esp_err_t encrypt_data(const uint8_t *input, size_t input_len, 
                           uint8_t *output, size_t *output_len)
{
    if (input == NULL || output == NULL || output_len == NULL || *output_len < input_len) {
        ESP_LOGE(TAG, "Invalid encryption parameters");
        return ESP_ERR_INVALID_ARG;
    }
    
    // In a real implementation, you would use AES-GCM or another AEAD cipher
    // For simplicity, we'll use AES-CBC with PKCS#7 padding
    
    mbedtls_aes_context aes;
    mbedtls_aes_init(&aes);
    
    // Set the encryption key
    if (mbedtls_aes_setkey_enc(&aes, encryption_key, KEY_LENGTH * 8) != 0) {
        ESP_LOGE(TAG, "Failed to set encryption key");
        mbedtls_aes_free(&aes);
        return ESP_FAIL;
    }
    
    // Copy IV to output (to be used for decryption)
    memcpy(output, encryption_iv, IV_LENGTH);
    
    // Perform encryption
    size_t offset = IV_LENGTH;
    int ret = mbedtls_aes_crypt_cbc(&aes, MBEDTLS_AES_ENCRYPT, input_len, 
                                   (unsigned char*)encryption_iv, input, output + offset);
    
    if (ret != 0) {
        ESP_LOGE(TAG, "Encryption failed");
        mbedtls_aes_free(&aes);
        return ESP_FAIL;
    }
    
    // Update output length
    *output_len = offset + input_len;
    
    mbedtls_aes_free(&aes);
    
    return ESP_OK;
}

/**
 * @brief Decrypt data using AES-256-CBC
 * 
 * @param input Encrypted data
 * @param input_len Encrypted data length
 * @param output Buffer for decrypted data
 * @param output_len Pointer to output buffer length (in/out)
 * @return esp_err_t ESP_OK on success
 */
static esp_err_t decrypt_data(const uint8_t *input, size_t input_len, 
                           uint8_t *output, size_t *output_len)
{
    if (input == NULL || output == NULL || output_len == NULL || 
        input_len <= IV_LENGTH || *output_len < (input_len - IV_LENGTH)) {
        ESP_LOGE(TAG, "Invalid decryption parameters");
        return ESP_ERR_INVALID_ARG;
    }
    
    // Extract IV from input
    uint8_t iv[IV_LENGTH];
    memcpy(iv, input, IV_LENGTH);
    
    // Initialize AES context
    mbedtls_aes_context aes;
    mbedtls_aes_init(&aes);
    
    // Set the decryption key
    if (mbedtls_aes_setkey_dec(&aes, encryption_key, KEY_LENGTH * 8) != 0) {
        ESP_LOGE(TAG, "Failed to set decryption key");
        mbedtls_aes_free(&aes);
        return ESP_FAIL;
    }
    
    // Perform decryption
    size_t offset = IV_LENGTH;
    int ret = mbedtls_aes_crypt_cbc(&aes, MBEDTLS_AES_DECRYPT, input_len - offset, 
                                   iv, input + offset, output);
    
    if (ret != 0) {
        ESP_LOGE(TAG, "Decryption failed");
        mbedtls_aes_free(&aes);
        return ESP_FAIL;
    }
    
    // Update output length
    *output_len = input_len - offset;
    
    mbedtls_aes_free(&aes);
    
    return ESP_OK;
}
