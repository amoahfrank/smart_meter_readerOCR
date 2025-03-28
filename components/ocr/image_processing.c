/**
 * @file image_processing.c
 * @brief Image processing functions for OCR preprocessing
 * 
 * This file implements various image processing functions to prepare
 * camera images for OCR, including binarization, noise reduction,
 * perspective correction, and digit segmentation.
 * 
 * @author Smart Meter Reader OCR Team
 * @date 2023
 */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "esp_camera.h"

// Include local headers
#include "image_processing.h"

static const char *TAG = "image_processing";

// Minimum dimensions for a valid digit
#define MIN_DIGIT_WIDTH 8
#define MIN_DIGIT_HEIGHT 12

// Default ROI (Region of Interest) percentages if no meter is detected
// These values assume the meter is roughly centered in the frame
#define DEFAULT_ROI_X_PERCENT 20
#define DEFAULT_ROI_Y_PERCENT 35
#define DEFAULT_ROI_WIDTH_PERCENT 60
#define DEFAULT_ROI_HEIGHT_PERCENT 30

// Forward declarations
static esp_err_t convert_to_grayscale(const camera_fb_t *fb, uint8_t **gray_image, size_t *gray_size);
static esp_err_t apply_adaptive_threshold(uint8_t *image, size_t width, size_t height);
static esp_err_t detect_meter_roi(const uint8_t *gray_image, size_t width, size_t height, 
                               int *roi_x, int *roi_y, int *roi_width, int *roi_height);
static esp_err_t extract_roi(const uint8_t *image, size_t width, size_t height, 
                          int roi_x, int roi_y, int roi_width, int roi_height,
                          uint8_t **roi_image);
static esp_err_t find_digit_bounding_boxes(const uint8_t *binary_image, size_t width, size_t height,
                                        digit_segments_t *segments);
static esp_err_t extract_digit(const uint8_t *image, size_t width, size_t height,
                            int x, int y, int w, int h, uint8_t **digit_image);
static void cleanup_binary_image(uint8_t *image, size_t width, size_t height);

/**
 * @brief Initialize the image processing module
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t image_processing_init(void)
{
    ESP_LOGI(TAG, "Initializing image processing module");
    
    // Nothing to initialize for now, but keeping the function for future use
    
    return ESP_OK;
}

/**
 * @brief Prepare an image for OCR processing
 * 
 * @param fb Input camera frame buffer
 * @param result Output structure with processed image and metadata
 * @return esp_err_t ESP_OK on success
 */
esp_err_t image_processing_prepare_for_ocr(const camera_fb_t *fb, image_processing_result_t *result)
{
    if (fb == NULL || result == NULL) {
        ESP_LOGE(TAG, "Invalid parameters");
        return ESP_ERR_INVALID_ARG;
    }
    
    ESP_LOGI(TAG, "Preparing image for OCR");
    
    // Start timing
    int64_t start_time = esp_timer_get_time();
    
    // Convert image to grayscale
    uint8_t *gray_image = NULL;
    size_t gray_width = fb->width;
    size_t gray_height = fb->height;
    size_t gray_size = 0;
    
    esp_err_t ret = convert_to_grayscale(fb, &gray_image, &gray_size);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to convert image to grayscale");
        return ret;
    }
    
    // Detect meter region of interest
    int roi_x, roi_y, roi_width, roi_height;
    ret = detect_meter_roi(gray_image, gray_width, gray_height, 
                         &roi_x, &roi_y, &roi_width, &roi_height);
    
    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "Could not detect meter ROI, using default");
        // Use default ROI values
        roi_x = (gray_width * DEFAULT_ROI_X_PERCENT) / 100;
        roi_y = (gray_height * DEFAULT_ROI_Y_PERCENT) / 100;
        roi_width = (gray_width * DEFAULT_ROI_WIDTH_PERCENT) / 100;
        roi_height = (gray_height * DEFAULT_ROI_HEIGHT_PERCENT) / 100;
    }
    
    // Extract ROI
    uint8_t *roi_image = NULL;
    ret = extract_roi(gray_image, gray_width, gray_height, 
                    roi_x, roi_y, roi_width, roi_height, 
                    &roi_image);
    
    // Free grayscale image, no longer needed
    free(gray_image);
    
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to extract ROI");
        return ret;
    }
    
    // Apply adaptive thresholding to the ROI
    ret = apply_adaptive_threshold(roi_image, roi_width, roi_height);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to apply threshold");
        free(roi_image);
        return ret;
    }
    
    // Clean up the binary image (remove noise, etc.)
    cleanup_binary_image(roi_image, roi_width, roi_height);
    
    // Fill in the result structure
    result->processed_image = roi_image;
    result->width = roi_width;
    result->height = roi_height;
    result->roi_x = roi_x;
    result->roi_y = roi_y;
    result->roi_width = roi_width;
    result->roi_height = roi_height;
    
    // Calculate processing time
    int64_t end_time = esp_timer_get_time();
    float processing_time_ms = (end_time - start_time) / 1000.0f;
    
    ESP_LOGI(TAG, "Image preprocessing completed in %.2f ms", processing_time_ms);
    
    return ESP_OK;
}

/**
 * @brief Segment digits from a preprocessed image
 * 
 * @param image Preprocessed binary image
 * @param image_size Size of the image in bytes
 * @param segments Output structure with digit segments
 * @return esp_err_t ESP_OK on success
 */
esp_err_t image_processing_segment_digits(const uint8_t *image, size_t image_size, 
                                        digit_segments_t *segments)
{
    if (image == NULL || segments == NULL || image_size == 0) {
        ESP_LOGE(TAG, "Invalid parameters for digit segmentation");
        return ESP_ERR_INVALID_ARG;
    }
    
    ESP_LOGI(TAG, "Segmenting digits from image");
    
    // Calculate image width and height (assuming square pixels)
    size_t width = (size_t)sqrt(image_size);
    size_t height = image_size / width;
    
    // Initialize segments
    segments->count = 0;
    for (int i = 0; i < MAX_DIGITS; i++) {
        segments->segments[i].image = NULL;
    }
    
    // Find bounding boxes for each digit
    esp_err_t ret = find_digit_bounding_boxes(image, width, height, segments);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to find digit bounding boxes");
        return ret;
    }
    
    // Extract each digit into its own image
    for (int i = 0; i < segments->count; i++) {
        int x = segments->segments[i].x;
        int y = segments->segments[i].y;
        int w = segments->segments[i].width;
        int h = segments->segments[i].height;
        
        uint8_t *digit_image = NULL;
        ret = extract_digit(image, width, height, x, y, w, h, &digit_image);
        
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "Failed to extract digit %d", i);
            // Clean up previously allocated segments
            for (int j = 0; j < i; j++) {
                if (segments->segments[j].image != NULL) {
                    free(segments->segments[j].image);
                    segments->segments[j].image = NULL;
                }
            }
            return ret;
        }
        
        segments->segments[i].image = digit_image;
    }
    
    // Sort segments from left to right (for reading order)
    // Simple bubble sort for simplicity
    for (int i = 0; i < segments->count - 1; i++) {
        for (int j = 0; j < segments->count - i - 1; j++) {
            if (segments->segments[j].x > segments->segments[j + 1].x) {
                // Swap segments
                digit_segment_t temp = segments->segments[j];
                segments->segments[j] = segments->segments[j + 1];
                segments->segments[j + 1] = temp;
            }
        }
    }
    
    ESP_LOGI(TAG, "Segmented %d digits", segments->count);
    
    return ESP_OK;
}

/**
 * @brief Resize a digit image to the target dimensions
 * 
 * @param digit_image Input digit image
 * @param width Input width
 * @param height Input height
 * @param resized_image Output resized image (must be pre-allocated)
 * @param target_width Target width
 * @param target_height Target height
 * @return esp_err_t ESP_OK on success
 */
esp_err_t image_processing_resize_digit(const uint8_t *digit_image, size_t width, size_t height,
                                     uint8_t *resized_image, size_t target_width, size_t target_height)
{
    if (digit_image == NULL || resized_image == NULL) {
        ESP_LOGE(TAG, "Invalid parameters for digit resize");
        return ESP_ERR_INVALID_ARG;
    }
    
    // Clear the target image
    memset(resized_image, 0, target_width * target_height);
    
    // Calculate scaling factors
    float scale_x = (float)width / target_width;
    float scale_y = (float)height / target_height;
    
    // Simple nearest neighbor scaling for simplicity
    // In a real application, consider bilinear interpolation for better quality
    for (size_t y = 0; y < target_height; y++) {
        for (size_t x = 0; x < target_width; x++) {
            size_t src_x = (size_t)(x * scale_x);
            size_t src_y = (size_t)(y * scale_y);
            
            if (src_x >= width) src_x = width - 1;
            if (src_y >= height) src_y = height - 1;
            
            resized_image[y * target_width + x] = digit_image[src_y * width + src_x];
        }
    }
    
    return ESP_OK;
}

/**
 * @brief Convert camera frame to grayscale
 * 
 * @param fb Input camera frame buffer
 * @param gray_image Output grayscale image (allocated by this function)
 * @param gray_size Output size of the grayscale image
 * @return esp_err_t ESP_OK on success
 */
static esp_err_t convert_to_grayscale(const camera_fb_t *fb, uint8_t **gray_image, size_t *gray_size)
{
    if (fb == NULL || gray_image == NULL || gray_size == NULL) {
        ESP_LOGE(TAG, "Invalid parameters for grayscale conversion");
        return ESP_ERR_INVALID_ARG;
    }
    
    size_t width = fb->width;
    size_t height = fb->height;
    size_t out_size = width * height;
    
    // Allocate memory for grayscale image
    uint8_t *out_buf = (uint8_t *)malloc(out_size);
    if (out_buf == NULL) {
        ESP_LOGE(TAG, "Failed to allocate memory for grayscale image");
        return ESP_ERR_NO_MEM;
    }
    
    // Process based on pixel format
    if (fb->format == PIXFORMAT_JPEG) {
        // For JPEG, we'd need to decode it first
        // This is a simplified placeholder - real implementation would need JPEG decoder
        ESP_LOGE(TAG, "JPEG format not supported for grayscale conversion");
        free(out_buf);
        return ESP_ERR_NOT_SUPPORTED;
    } else if (fb->format == PIXFORMAT_RGB565) {
        // Convert RGB565 to grayscale
        for (size_t i = 0; i < height; i++) {
            for (size_t j = 0; j < width; j++) {
                size_t src_idx = (i * width + j) * 2; // 2 bytes per RGB565 pixel
                size_t dst_idx = i * width + j;
                
                // Extract RGB components from RGB565
                uint16_t pixel = (fb->buf[src_idx + 1] << 8) | fb->buf[src_idx];
                uint8_t r = (pixel >> 11) & 0x1F;
                uint8_t g = (pixel >> 5) & 0x3F;
                uint8_t b = pixel & 0x1F;
                
                // Scale to 0-255 range
                r = (r * 255) / 31;
                g = (g * 255) / 63;
                b = (b * 255) / 31;
                
                // Convert to grayscale using luminance formula
                out_buf[dst_idx] = (uint8_t)(0.299 * r + 0.587 * g + 0.114 * b);
            }
        }
    } else if (fb->format == PIXFORMAT_GRAYSCALE) {
        // Already grayscale, just copy
        memcpy(out_buf, fb->buf, out_size);
    } else {
        // Unsupported format
        ESP_LOGE(TAG, "Unsupported pixel format for grayscale conversion");
        free(out_buf);
        return ESP_ERR_NOT_SUPPORTED;
    }
    
    *gray_image = out_buf;
    *gray_size = out_size;
    
    return ESP_OK;
}

/**
 * @brief Apply adaptive thresholding to an image
 * 
 * @param image Image to process (in-place)
 * @param width Image width
 * @param height Image height
 * @return esp_err_t ESP_OK on success
 */
static esp_err_t apply_adaptive_threshold(uint8_t *image, size_t width, size_t height)
{
    if (image == NULL) {
        ESP_LOGE(TAG, "Invalid image for thresholding");
        return ESP_ERR_INVALID_ARG;
    }
    
    const int block_size = 11;  // Size of neighborhood for adaptive threshold
    const int c = 7;            // Constant subtracted from the mean
    
    // Allocate temporary buffer for the result
    uint8_t *result = (uint8_t *)malloc(width * height);
    if (result == NULL) {
        ESP_LOGE(TAG, "Failed to allocate memory for thresholding");
        return ESP_ERR_NO_MEM;
    }
    
    // Apply adaptive thresholding
    int half_block = block_size / 2;
    
    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            // Calculate local mean
            int sum = 0;
            int count = 0;
            
            for (int dy = -half_block; dy <= half_block; dy++) {
                for (int dx = -half_block; dx <= half_block; dx++) {
                    int ny = y + dy;
                    int nx = x + dx;
                    
                    if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                        sum += image[ny * width + nx];
                        count++;
                    }
                }
            }
            
            uint8_t mean = (count > 0) ? (sum / count) : 0;
            uint8_t threshold = (mean > c) ? (mean - c) : 0;
            
            // Apply threshold
            result[y * width + x] = (image[y * width + x] < threshold) ? 0 : 255;
        }
    }
    
    // Copy result back to the input image
    memcpy(image, result, width * height);
    free(result);
    
    return ESP_OK;
}

/**
 * @brief Detect the meter region of interest in an image
 * 
 * @param gray_image Grayscale input image
 * @param width Image width
 * @param height Image height
 * @param roi_x Output ROI x coordinate
 * @param roi_y Output ROI y coordinate
 * @param roi_width Output ROI width
 * @param roi_height Output ROI height
 * @return esp_err_t ESP_OK on success
 */
static esp_err_t detect_meter_roi(const uint8_t *gray_image, size_t width, size_t height, 
                                int *roi_x, int *roi_y, int *roi_width, int *roi_height)
{
    // This is a simplified implementation that assumes the meter is in the center of the image
    // A more robust implementation would use edge detection, contour finding, etc.
    
    // For now, just return a centered ROI of fixed size
    *roi_x = (width * DEFAULT_ROI_X_PERCENT) / 100;
    *roi_y = (height * DEFAULT_ROI_Y_PERCENT) / 100;
    *roi_width = (width * DEFAULT_ROI_WIDTH_PERCENT) / 100;
    *roi_height = (height * DEFAULT_ROI_HEIGHT_PERCENT) / 100;
    
    // Make sure ROI is within image bounds
    if (*roi_x + *roi_width > width) {
        *roi_width = width - *roi_x;
    }
    
    if (*roi_y + *roi_height > height) {
        *roi_height = height - *roi_y;
    }
    
    return ESP_OK;
}

/**
 * @brief Extract a region of interest from an image
 * 
 * @param image Input image
 * @param width Image width
 * @param height Image height
 * @param roi_x ROI x coordinate
 * @param roi_y ROI y coordinate
 * @param roi_width ROI width
 * @param roi_height ROI height
 * @param roi_image Output ROI image (allocated by this function)
 * @return esp_err_t ESP_OK on success
 */
static esp_err_t extract_roi(const uint8_t *image, size_t width, size_t height, 
                          int roi_x, int roi_y, int roi_width, int roi_height,
                          uint8_t **roi_image)
{
    if (image == NULL || roi_image == NULL) {
        ESP_LOGE(TAG, "Invalid parameters for ROI extraction");
        return ESP_ERR_INVALID_ARG;
    }
    
    // Check if ROI is within image bounds
    if (roi_x < 0 || roi_y < 0 || roi_x + roi_width > width || roi_y + roi_height > height) {
        ESP_LOGE(TAG, "ROI is outside image bounds");
        return ESP_ERR_INVALID_ARG;
    }
    
    // Allocate memory for ROI
    uint8_t *out_buf = (uint8_t *)malloc(roi_width * roi_height);
    if (out_buf == NULL) {
        ESP_LOGE(TAG, "Failed to allocate memory for ROI");
        return ESP_ERR_NO_MEM;
    }
    
    // Extract ROI
    for (int y = 0; y < roi_height; y++) {
        for (int x = 0; x < roi_width; x++) {
            int src_idx = (roi_y + y) * width + (roi_x + x);
            int dst_idx = y * roi_width + x;
            out_buf[dst_idx] = image[src_idx];
        }
    }
    
    *roi_image = out_buf;
    
    return ESP_OK;
}

/**
 * @brief Find bounding boxes for digits in a binary image
 * 
 * @param binary_image Binary input image
 * @param width Image width
 * @param height Image height
 * @param segments Output structure with digit segments
 * @return esp_err_t ESP_OK on success
 */
static esp_err_t find_digit_bounding_boxes(const uint8_t *binary_image, size_t width, size_t height,
                                        digit_segments_t *segments)
{
    if (binary_image == NULL || segments == NULL) {
        ESP_LOGE(TAG, "Invalid parameters for finding digit bounding boxes");
        return ESP_ERR_INVALID_ARG;
    }
    
    // Create a label image for connected component analysis
    int *labels = (int *)calloc(width * height, sizeof(int));
    if (labels == NULL) {
        ESP_LOGE(TAG, "Failed to allocate memory for labels");
        return ESP_ERR_NO_MEM;
    }
    
    // Connected component analysis (simple two-pass algorithm)
    int next_label = 1;
    int *equiv = (int *)calloc(width * height / 4, sizeof(int)); // Equivalence table
    
    // First pass: assign initial labels and record equivalences
    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            // Skip background pixels
            if (binary_image[y * width + x] == 0) {
                continue;
            }
            
            // Check neighbors (4-connected)
            int up = (y > 0) ? labels[(y - 1) * width + x] : 0;
            int left = (x > 0) ? labels[y * width + (x - 1)] : 0;
            
            if (up == 0 && left == 0) {
                // New label
                labels[y * width + x] = next_label;
                equiv[next_label] = next_label;
                next_label++;
            } else if (up != 0 && left == 0) {
                // Copy from above
                labels[y * width + x] = up;
            } else if (up == 0 && left != 0) {
                // Copy from left
                labels[y * width + x] = left;
            } else {
                // Both neighbors have labels, use the smaller one and record equivalence
                int min_label = (up < left) ? up : left;
                int max_label = (up > left) ? up : left;
                labels[y * width + x] = min_label;
                
                // Record equivalence
                if (equiv[max_label] != min_label) {
                    int old_equiv = equiv[max_label];
                    for (int i = 1; i < next_label; i++) {
                        if (equiv[i] == old_equiv) {
                            equiv[i] = equiv[min_label];
                        }
                    }
                }
            }
        }
    }
    
    // Resolve equivalences to smallest label
    for (int i = 1; i < next_label; i++) {
        int label = i;
        while (equiv[label] != label) {
            label = equiv[label];
        }
        equiv[i] = label;
    }
    
    // Second pass: update labels
    for (size_t i = 0; i < width * height; i++) {
        if (labels[i] != 0) {
            labels[i] = equiv[labels[i]];
        }
    }
    
    // Find bounding boxes for each label
    int max_components = 50; // Maximum number of components to consider
    int *x_min = (int *)calloc(max_components, sizeof(int));
    int *y_min = (int *)calloc(max_components, sizeof(int));
    int *x_max = (int *)calloc(max_components, sizeof(int));
    int *y_max = (int *)calloc(max_components, sizeof(int));
    int *pixel_count = (int *)calloc(max_components, sizeof(int));
    
    if (!x_min || !y_min || !x_max || !y_max || !pixel_count) {
        ESP_LOGE(TAG, "Failed to allocate memory for bounding boxes");
        free(labels);
        free(equiv);
        free(x_min);
        free(y_min);
        free(x_max);
        free(y_max);
        free(pixel_count);
        return ESP_ERR_NO_MEM;
    }
    
    // Initialize bounds
    for (int i = 0; i < max_components; i++) {
        x_min[i] = width;
        y_min[i] = height;
        x_max[i] = 0;
        y_max[i] = 0;
    }
    
    // Find bounding boxes
    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            int label = labels[y * width + x];
            if (label == 0 || label >= max_components) {
                continue;
            }
            
            if (x < x_min[label]) x_min[label] = x;
            if (y < y_min[label]) y_min[label] = y;
            if (x > x_max[label]) x_max[label] = x;
            if (y > y_max[label]) y_max[label] = y;
            pixel_count[label]++;
        }
    }
    
    // Count valid components and filter by size
    int valid_count = 0;
    for (int i = 1; i < next_label && i < max_components; i++) {
        int width = x_max[i] - x_min[i] + 1;
        int height = y_max[i] - y_min[i] + 1;
        
        // Filter small components (likely noise)
        if (width >= MIN_DIGIT_WIDTH && height >= MIN_DIGIT_HEIGHT && 
            pixel_count[i] > MIN_DIGIT_WIDTH * MIN_DIGIT_HEIGHT / 2) {
            valid_count++;
        }
    }
    
    // Limit to max number of digits
    if (valid_count > MAX_DIGITS) {
        ESP_LOGW(TAG, "Found %d digits, limiting to %d", valid_count, MAX_DIGITS);
        valid_count = MAX_DIGITS;
    }
    
    // Fill segments structure
    segments->count = 0;
    for (int i = 1; i < next_label && i < max_components && segments->count < valid_count; i++) {
        int width = x_max[i] - x_min[i] + 1;
        int height = y_max[i] - y_min[i] + 1;
        
        // Filter small components (likely noise)
        if (width >= MIN_DIGIT_WIDTH && height >= MIN_DIGIT_HEIGHT && 
            pixel_count[i] > MIN_DIGIT_WIDTH * MIN_DIGIT_HEIGHT / 2) {
            segments->segments[segments->count].x = x_min[i];
            segments->segments[segments->count].y = y_min[i];
            segments->segments[segments->count].width = width;
            segments->segments[segments->count].height = height;
            segments->count++;
        }
    }
    
    // Clean up
    free(labels);
    free(equiv);
    free(x_min);
    free(y_min);
    free(x_max);
    free(y_max);
    free(pixel_count);
    
    return ESP_OK;
}

/**
 * @brief Extract a digit from an image
 * 
 * @param image Input image
 * @param width Image width
 * @param height Image height
 * @param x Digit bounding box x coordinate
 * @param y Digit bounding box y coordinate
 * @param w Digit bounding box width
 * @param h Digit bounding box height
 * @param digit_image Output digit image (allocated by this function)
 * @return esp_err_t ESP_OK on success
 */
static esp_err_t extract_digit(const uint8_t *image, size_t width, size_t height,
                            int x, int y, int w, int h, uint8_t **digit_image)
{
    if (image == NULL || digit_image == NULL) {
        ESP_LOGE(TAG, "Invalid parameters for digit extraction");
        return ESP_ERR_INVALID_ARG;
    }
    
    // Check bounds
    if (x < 0 || y < 0 || x + w > width || y + h > height) {
        ESP_LOGE(TAG, "Digit bounding box out of bounds");
        return ESP_ERR_INVALID_ARG;
    }
    
    // Allocate memory for digit
    uint8_t *out_buf = (uint8_t *)malloc(w * h);
    if (out_buf == NULL) {
        ESP_LOGE(TAG, "Failed to allocate memory for digit");
        return ESP_ERR_NO_MEM;
    }
    
    // Extract digit
    for (int dy = 0; dy < h; dy++) {
        for (int dx = 0; dx < w; dx++) {
            int src_idx = (y + dy) * width + (x + dx);
            int dst_idx = dy * w + dx;
            out_buf[dst_idx] = image[src_idx];
        }
    }
    
    *digit_image = out_buf;
    
    return ESP_OK;
}

/**
 * @brief Clean up a binary image by removing noise
 * 
 * @param image Binary image to clean (in-place)
 * @param width Image width
 * @param height Image height
 */
static void cleanup_binary_image(uint8_t *image, size_t width, size_t height)
{
    if (image == NULL) {
        return;
    }
    
    // Simple noise removal - remove isolated pixels
    uint8_t *temp = (uint8_t *)malloc(width * height);
    if (temp == NULL) {
        ESP_LOGE(TAG, "Failed to allocate memory for image cleanup");
        return;
    }
    
    // Copy input to temp
    memcpy(temp, image, width * height);
    
    // Remove isolated foreground pixels
    for (size_t y = 1; y < height - 1; y++) {
        for (size_t x = 1; x < width - 1; x++) {
            int idx = y * width + x;
            
            // Skip background pixels
            if (temp[idx] == 0) {
                continue;
            }
            
            // Count foreground neighbors
            int count = 0;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    if (temp[(y + dy) * width + (x + dx)] != 0) {
                        count++;
                    }
                }
            }
            
            // Remove isolated pixels (less than 2 neighbors)
            if (count < 2) {
                image[idx] = 0;
            }
        }
    }
    
    // Copy back to temp for next operation
    memcpy(temp, image, width * height);
    
    // Fill small holes
    for (size_t y = 1; y < height - 1; y++) {
        for (size_t x = 1; x < width - 1; x++) {
            int idx = y * width + x;
            
            // Skip foreground pixels
            if (temp[idx] != 0) {
                continue;
            }
            
            // Count background neighbors
            int count = 0;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    if (temp[(y + dy) * width + (x + dx)] == 0) {
                        count++;
                    }
                }
            }
            
            // Fill small holes (less than 5 background neighbors)
            if (count < 5) {
                image[idx] = 255;
            }
        }
    }
    
    free(temp);
}

/**
 * @brief Deinitialize the image processing module
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t image_processing_deinit(void)
{
    ESP_LOGI(TAG, "Deinitializing image processing module");
    
    // Nothing to deinitialize for now, but keeping the function for future use
    
    return ESP_OK;
}
