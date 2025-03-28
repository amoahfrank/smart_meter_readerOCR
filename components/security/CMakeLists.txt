set(COMPONENT_SRCS
    "security_manager.c"
    "ota_manager.c"
)

set(COMPONENT_ADD_INCLUDEDIRS
    "include"
)

set(COMPONENT_REQUIRES
    esp_common
    freertos
    mbedtls
    nvs_flash
    esp_https_ota
    app_update
    esp_crt_bundle
)

register_component()
