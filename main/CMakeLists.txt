set(COMPONENT_SRCS
    "app_main.c"
    "state_machine.c"
)

set(COMPONENT_ADD_INCLUDEDIRS
    "include"
)

set(COMPONENT_REQUIRES
    camera
    display
    ocr
    connectivity
    power_mgmt
    security
    configuration
)

register_component()
