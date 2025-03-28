cmake_minimum_required(VERSION 3.5)

set(PROJECT_NAME "smart_meter_readerOCR")
set(PROJECT_VER "1.0.0")

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(${PROJECT_NAME})

# Add the model binary to the firmware
target_add_binary_data(${PROJECT_NAME}.elf "models/digit_recognition_model.tflite" TEXT)

# Add the root CA certificate for OTA and HTTPS communication
target_add_binary_data(${PROJECT_NAME}.elf "certs/ca_cert.pem" TEXT)

# Set compiler warnings
target_compile_options(${PROJECT_NAME}.elf PRIVATE -Wall -Werror=all -Wextra -Wno-error=deprecated-declarations)

# Create a custom target to build and flash the firmware
add_custom_target(flash
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target flash
    DEPENDS ${PROJECT_NAME}.elf
    COMMENT "Flashing ${PROJECT_NAME} to target device"
)

# Create a custom target to build, flash, and monitor
add_custom_target(flash_monitor
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target flash_monitor
    DEPENDS ${PROJECT_NAME}.elf
    COMMENT "Flashing ${PROJECT_NAME} to target device and starting monitoring"
)

# Create a custom target to build and monitor
add_custom_target(monitor
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target monitor
    DEPENDS ${PROJECT_NAME}.elf
    COMMENT "Starting monitor for ${PROJECT_NAME}"
)

# Create a custom target to generate documentation
add_custom_target(docs
    COMMAND echo "Generating documentation for ${PROJECT_NAME}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    COMMENT "Generating documentation"
)

# Configuration message
message(STATUS "Building ${PROJECT_NAME} version ${PROJECT_VER}")
message(STATUS "IDF_PATH: $ENV{IDF_PATH}")
message(STATUS "Project path: ${PROJECT_SOURCE_DIR}")
