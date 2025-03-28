set(COMPONENT_SRCS
    "configuration.c"
)

set(COMPONENT_ADD_INCLUDEDIRS
    "include"
)

set(COMPONENT_REQUIRES
    esp_common
    freertos
    nvs_flash
    security
)

register_component()
