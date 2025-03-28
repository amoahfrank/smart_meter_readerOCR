set(COMPONENT_SRCS
    "camera.c"
)

set(COMPONENT_ADD_INCLUDEDIRS
    "include"
)

set(COMPONENT_REQUIRES
    esp32-camera
    esp_common
    freertos
)

register_component()
