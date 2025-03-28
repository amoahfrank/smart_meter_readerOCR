set(COMPONENT_SRCS
    "ocr.c"
    "image_processing.c"
    "model_interface.c"
)

set(COMPONENT_ADD_INCLUDEDIRS
    "include"
)

set(COMPONENT_REQUIRES
    esp32-camera
    esp_common
    freertos
    tensorflow-micro
)

register_component()
