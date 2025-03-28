set(COMPONENT_SRCS
    "power_mgmt.c"
)

set(COMPONENT_ADD_INCLUDEDIRS
    "include"
)

set(COMPONENT_REQUIRES
    esp_common
    freertos
    esp_adc
    driver
)

register_component()
