set(COMPONENT_SRCS
    "wifi_manager.c"
    "ble_manager.c"
    "lora_manager.c"
)

set(COMPONENT_ADD_INCLUDEDIRS
    "include"
)

set(COMPONENT_REQUIRES
    esp_common
    freertos
    esp_wifi
    bt
    nvs_flash
    mbedtls
    esp_http_client
    security
)

register_component()
