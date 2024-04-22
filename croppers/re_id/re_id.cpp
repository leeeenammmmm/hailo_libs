/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/
#include <vector>
#include <iostream>
#include "re_id.hpp"

#define PERSON_LABEL "person"
#define MIN_RATIO (1.7f)
#define MAX_RATIO (4.5f)
#define MIN_HEIGHT (0.3f)
#define MAX_HEIGHT (0.8f)
#define MIN_X (0.05f)
#define MAX_X (0.95f)
#define TRACK_DELAY (5)
#define MIN_QUALITY (400)
#define RE_ID_NETWORK_SIZE (cv::Size(128, 256))
std::map<int, int> track_counter;

cv::Mat convertNV12toGray(const cv::Mat& nv12Mat, int width, int height) {
    cv::Mat grayMat(height, width, CV_8UC1);
    
    // Extract the Y component (luminance) from the NV12 Mat
    int ySize = width * height;
    cv::Mat yChannel = nv12Mat(cv::Rect(0, 0, width, height));
    
    // Copy Y component to grayscale Mat
    yChannel.copyTo(grayMat);

    return grayMat;
}



/**
 * @brief Returns the quaility estimation of the person's crop.
 *
 * @param image  -  cv::Mat
 *        The original image.
 *
 * @param roi  -  HailoBBox
 *        The Bounding box of the person to calculate quality estimation on.
 *
 * @return float
 *         The quality estimation of the person.
 */
float quality_estimation(const cv::Mat &image, const HailoBBox &roi)
{
    // Crop the center of the roi from the image, avoid cropping out of bounds
    int cropped_xmin = CLAMP((image.cols * roi.xmin()), 0, image.cols);
    int cropped_ymin = CLAMP((image.rows * roi.ymin()), 0, image.rows);
    int cropped_xmax = CLAMP((image.cols * roi.xmax()), cropped_xmin, image.cols);
    int cropped_ymax = CLAMP((image.rows * roi.ymax()), cropped_ymin, image.rows);
    int cropped_width = cropped_xmax - cropped_xmin;
    int cropped_height = cropped_ymax - cropped_ymin;

    // If it is not too small then we can make the crop
    cv::Rect center_crop(cropped_xmin, cropped_ymin, cropped_width, cropped_height);
    cv::Mat cropped_image = image(center_crop);

    // Resize the frame
    cv::Mat resized_image;
    cv::resize(cropped_image, resized_image, RE_ID_NETWORK_SIZE, 0, 0, cv::INTER_LINEAR);

    // Convert to grayscale
    cv::Mat gray_image;
    cv::cvtColor(resized_image, gray_image, cv::COLOR_RGB2GRAY);

    // Compute the Laplacian of the gray image
    cv::Mat laplacian_image;
    cv::Laplacian(gray_image, laplacian_image, CV_64F);

    // Calculate the quality of person
    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacian_image, mean, stddev, cv::Mat());
    float quality = stddev.val[0] * stddev.val[0];

    // Release resources
    resized_image.release();
    cropped_image.release();
    gray_image.release();
    laplacian_image.release();

    return quality;
}

float quality_estimation_nv12(std::shared_ptr<HailoMat> hailo_mat, const HailoBBox &roi, const float crop_ratio = 1)
{
    // Crop the center of the roi from the image, avoid cropping out of bounds
    float roi_width = roi.width();
    float roi_height = roi.height();
    float roi_xmin = roi.xmin();
    float roi_ymin = roi.ymin();
    float roi_xmax = roi.xmax();
    float roi_ymax = roi.ymax();
    float x_offset = roi_width * crop_ratio;
    float y_offset = roi_height * crop_ratio;
    float cropped_xmin = CLAMP(roi_xmin + x_offset, 0, 1);
    float cropped_ymin = CLAMP(roi_ymin + y_offset, 0, 1);
    float cropped_xmax = CLAMP(roi_xmax - x_offset, cropped_xmin, 1);
    float cropped_ymax = CLAMP(roi_ymax - y_offset, cropped_ymin, 1);
    float cropped_width_n = cropped_xmax - cropped_xmin;
    float cropped_height_n = cropped_ymax - cropped_ymin;
    int cropped_width = int(cropped_width_n * hailo_mat->native_width());
    int cropped_height = int(cropped_height_n * hailo_mat->native_height());

    // If the cropepd image is too small then quality is zero
    if (cropped_width <= 10 || cropped_height <= 10)
        return -1.0;

    // If it is not too small then we can make the crop
    HailoROIPtr crop_roi = std::make_shared<HailoROI>(HailoBBox(cropped_xmin, cropped_ymin, cropped_width_n, cropped_height_n));
    std::vector<cv::Mat> cropped_image_vec = hailo_mat->crop(crop_roi);

    // Convert image to BGR
    cv::Mat bgr_image;
    switch (hailo_mat->get_type())
    {
    case HAILO_MAT_YUY2:
    {
        cv::Mat cropped_image = cropped_image_vec[0];
        cv::Mat yuy2_image = cv::Mat(cropped_image.rows, cropped_image.cols * 2, CV_8UC2, (char *)cropped_image.data, cropped_image.step);
        cv::cvtColor(yuy2_image, bgr_image, cv::COLOR_YUV2BGR_YUY2);
        break;
    }
    case HAILO_MAT_NV12:
    {
        std::cout << "convert nv12 to bgr!" << std::endl;
        cv::Mat full_mat = cv::Mat(cropped_image_vec[0].rows + cropped_image_vec[1].rows, cropped_image_vec[0].cols, CV_8UC1);
        memcpy(full_mat.data, cropped_image_vec[0].data, cropped_image_vec[0].rows * cropped_image_vec[0].cols);
        memcpy(full_mat.data + cropped_image_vec[0].rows * cropped_image_vec[0].cols, cropped_image_vec[1].data, cropped_image_vec[1].rows * cropped_image_vec[1].cols);
        cv::cvtColor(full_mat, bgr_image, cv::COLOR_YUV2BGR_NV12);
        std::cout << "bug in nv12!" << std::endl;

        break;
    }
    default:
        bgr_image = cropped_image_vec[0];
        break;
    }

    // Resize the frame
    cv::Mat resized_image;
    cv::resize(bgr_image, resized_image, RE_ID_NETWORK_SIZE, 0, 0, cv::INTER_LINEAR);

    // Convert to grayscale
    // cv::Mat gray_image = convertNV12toGray(resized_image, 128, 256);
    cv::Mat gray_image;
    std::cout << "Maybe bug here!" << std::endl;
    cv::cvtColor(resized_image, gray_image, cv::COLOR_BGR2GRAY);
    std::cout << "Here is not bug!" << std::endl;

    // Compute the Laplacian of the gray image
    cv::Mat laplacian_image;
    cv::Laplacian(gray_image, laplacian_image, CV_64F);

    // Calculate the quality of person
    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacian_image, mean, stddev, cv::Mat());
    float quality = stddev.val[0] * stddev.val[0];

    // Release resources
    resized_image.release();
    gray_image.release();
    laplacian_image.release();

    return quality;
}

HailoUniqueIDPtr get_tracking_id(HailoDetectionPtr detection)
{
    for (auto obj : detection->get_objects_typed(HAILO_UNIQUE_ID))
    {
        HailoUniqueIDPtr id = std::dynamic_pointer_cast<HailoUniqueID>(obj);
        if (id->get_mode() == TRACKING_ID)
        {
            return id;
        }
    }
    return nullptr;
}

/**
 * @brief Returns a vector of HailoROIPtr to crop and resize.
 *
 * @param image The original picture (cv::Mat).
 * @param roi The main ROI of this picture.
 * @return std::vector<HailoROIPtr> vector of ROI's to crop and resize.
 */
std::vector<HailoROIPtr> create_crops(std::shared_ptr<HailoMat> image, HailoROIPtr roi)
{
    std::vector<HailoROIPtr> crop_rois;
    // Get all detections.
    std::vector<HailoDetectionPtr> detections_ptrs = hailo_common::get_hailo_detections(roi);
    for (HailoDetectionPtr &detection : detections_ptrs)
    {
        // Modify only detections with "person" label.
        if (std::string(PERSON_LABEL) == detection->get_label())
        {
            // Remove previous matrices
            roi->remove_objects_typed(HAILO_MATRIX);

            int tracking_id = get_tracking_id(detection)->get_id();

            auto counter = track_counter.find(tracking_id);
            if (counter == track_counter.end())
            {
                track_counter[tracking_id] = 0;
            }
            else if (counter->second < TRACK_DELAY)
            {
                track_counter[tracking_id] += 1;
            }
            else
            {
                auto bbox = detection->get_bbox();
                float quality = quality_estimation_nv12(image, bbox);
                // float quality = 500;
                float ratio = (bbox.height() * image->height()) / (bbox.width() * image->width());
                if (ratio > MIN_RATIO && ratio < MAX_RATIO &&
                    bbox.height() > MIN_HEIGHT && bbox.height() < MAX_HEIGHT &&
                    bbox.xmin() > MIN_X && bbox.xmax() < MAX_X && quality > MIN_QUALITY)
                {
                    crop_rois.emplace_back(detection);
                }
            }
        }
    }
    return crop_rois;
}
