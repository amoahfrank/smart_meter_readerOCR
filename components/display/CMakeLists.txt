set(COMPONENT_SRCS
    "display.c"
)

set(COMPONENT_ADD_INCLUDEDIRS
    "include"
)

set(COMPONENT_REQUIRES
    epaper
    esp_common
    freertos
)

register_component()
