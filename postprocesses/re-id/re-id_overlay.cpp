/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/

#include <gst/video/video-format.h>
#include <gst/gst.h>
#include <gst/gstbuffer.h>
#include <gst/video/video.h>
#include <iostream>
#include <map>
#include <typeinfo>
#include <math.h>

// Hailo includes
#include "re-id_overlay.hpp"
#include "hailo_common.hpp"

// Open source includes
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>

// General
#define TEXT_THICKNESS (1)
#define TEXT_CLS_THICKNESS (2)
#define TEXT_FONT_FACTOR (0.12f)
#define DEFAULT_DETECTION_COLOR (cv::Scalar(255, 255, 255))

static const std::vector<cv::Scalar> color_table = {
    cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255), cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 255),
    cv::Scalar(255, 0, 255), cv::Scalar(255, 170, 0), cv::Scalar(255, 0, 170), cv::Scalar(0, 255, 170), cv::Scalar(170, 255, 0),
    cv::Scalar(170, 0, 255), cv::Scalar(0, 170, 255), cv::Scalar(255, 85, 0), cv::Scalar(85, 255, 0), cv::Scalar(0, 255, 85),
    cv::Scalar(0, 85, 255), cv::Scalar(85, 0, 255), cv::Scalar(255, 0, 85)};

cv::Scalar indexToColor(size_t index)
{
    return color_table[index % color_table.size()];
}

HailoUniqueIDPtr get_global_id(HailoDetectionPtr detection)
{
    for (auto obj : detection->get_objects_typed(HAILO_UNIQUE_ID))
    {
        HailoUniqueIDPtr id = std::dynamic_pointer_cast<HailoUniqueID>(obj);
        if (id->get_mode() == GLOBAL_ID)
        {
            return id;
        }
    }
    return nullptr;
}

// void face_blur(cv::Mat &mat, HailoROIPtr roi)
// {
//     for (auto detection : hailo_common::get_hailo_detections(roi))
//     {
//         if (detection->get_label() == "face")
//         {
//             HailoBBox roi_bbox = hailo_common::create_flattened_bbox(roi->get_bbox(), roi->get_scaling_bbox());
//             auto detection_bbox = detection->get_bbox();
//             auto xmin = std::clamp<int>(((detection_bbox.xmin() * roi_bbox.width()) + roi_bbox.xmin()) * mat.cols, 0, mat.cols);
//             auto ymin = std::clamp<int>(((detection_bbox.ymin() * roi_bbox.height()) + roi_bbox.ymin()) * mat.rows, 0, mat.rows);
//             auto xmax = std::clamp<int>(((detection_bbox.xmax() * roi_bbox.width()) + roi_bbox.xmin()) * mat.cols, 0, mat.cols);
//             auto ymax = std::clamp<int>(((detection_bbox.ymax() * roi_bbox.height()) + roi_bbox.ymin()) * mat.rows, 0, mat.rows);
//             auto bbox_min = cv::Point(xmin, ymin);
//             auto bbox_max = cv::Point(xmax, ymax);
//             auto rect = cv::Rect(bbox_min, bbox_max);
//             cv::Mat face = mat(rect);
//             cv::blur(face, face, cv::Size(11, 11));
//             cv::rectangle(mat, bbox_min, bbox_max, cv::Scalar(0,0,0), 2);
//             roi->remove_object(detection);
//         }
//         else
//         {
//             face_blur(mat, detection);
//         }
//     }
// }

static void printImageDimensions(const cv::Mat& image) {
    int dimensions = image.dims;
    const int* size = image.size;

    std::cout << "Image dimensions: " << dimensions << " (";

    for (int i = 0; i < dimensions; ++i) {
        std::cout << size[i];
        if (i < dimensions - 1) {
            std::cout << " x ";
        }
    }

    std::cout << ")" << std::endl;
}

static cv::Mat bgrToNV12(const cv::Mat& bgrImage) {
    // Step 1: Split BGR image into channels
    cv::Mat channels[3];
    cv::split(bgrImage, channels);

    // Step 2: Convert to grayscale (Y plane)
    cv::Mat yPlane;
    cv::cvtColor(bgrImage, yPlane, cv::COLOR_BGR2GRAY);

    // Step 3: Resize and combine blue and red channels to form UV plane
    cv::Mat uvPlane;
    cv::resize(channels[0], channels[0], cv::Size(), 0.5, 0.5); // Downsample blue channel
    cv::resize(channels[2], channels[2], cv::Size(), 0.5, 0.5); // Downsample red channel
    std::vector<cv::Mat> uvChannels = {channels[0], channels[2]};
    cv::merge(uvChannels, uvPlane); // Combine blue and red channels
    std::cout << "Bug here" << std::endl;
    // Step 4: Concatenate Y and UV planes to form NV12 image
    std::vector<cv::Mat> nv12Planes = {yPlane, uvPlane};
    cv::Mat nv12Image;
    cv::merge(nv12Planes, nv12Image);

    return nv12Image;
}

static cv::Mat BGR2YUV_NV12(const cv::Mat &src) {
  auto src_h = src.rows, src_w = src.cols;
  cv::Mat dst(src_h * 1.5, src_w, CV_8UC1);
  cv::cvtColor(src, dst, cv::COLOR_BGR2YUV_I420);
  auto n_y = src_h * src_w;
  auto n_uv = n_y / 2, n_u = n_y / 4;
  std::vector<uint8_t> uv(n_uv);
  std::copy(dst.data+n_y, dst.data+n_y+n_uv, uv.data());
  for (auto i = 0; i < n_u; i++) {
    dst.data[n_y + 2*i] = uv[i];            // U
    dst.data[n_y + 2*i + 1] = uv[n_u + i];  // V
  }
  return dst;
}

static void draw_detection(cv::Mat &image_planes, HailoDetectionPtr detection, HailoROIPtr roi, int font_thickness = 1, int line_thickness = 1)
{

    HailoBBox roi_bbox = hailo_common::create_flattened_bbox(roi->get_bbox(), roi->get_scaling_bbox());
    auto detection_bbox = detection->get_bbox();
    auto global_id = get_global_id(detection);

    auto bbox_min = cv::Point(((detection_bbox.xmin() * roi_bbox.width()) + roi_bbox.xmin()) * image_planes.cols,
                              ((detection_bbox.ymin() * roi_bbox.height()) + roi_bbox.ymin()) * image_planes.rows);
    auto bbox_max = cv::Point(((detection_bbox.xmax() * roi_bbox.width()) + roi_bbox.xmin()) * image_planes.cols,
                              ((detection_bbox.ymax() * roi_bbox.height()) + roi_bbox.ymin()) * image_planes.rows);
    auto bbox_width = bbox_max.x - bbox_min.x;

    // Draw the detection box
    auto color_rgb = (!global_id) ? DEFAULT_DETECTION_COLOR : indexToColor(global_id->get_id());
    cv::rectangle(image_planes, bbox_min, bbox_max, color_rgb, line_thickness);


    if (global_id)
    {
        std::string id_text = std::to_string(global_id->get_id());
        // Calculating the font size according to the box width.
        float font_scale = TEXT_FONT_FACTOR * log(bbox_width);
        
        auto text_position = cv::Point(bbox_min.x + log(bbox_width), bbox_max.y - log(bbox_width));
        
        // Draw the class and confidence text
        cv::putText(image_planes, id_text, text_position, cv::FONT_HERSHEY_SIMPLEX, font_scale, color_rgb, font_thickness);
        std::cout << "Detect an old person has id: " << id_text << std::endl;
    }
    else{
        std::cout << "Detect a new person!" << std::endl;
    }
}

static void copyNv12DataToGstVideoFrame(const cv::Mat& nv12Image, GstVideoFrame* frame) {
    // Assuming nv12Image is a cv::Mat representing an NV12 image

    // Map the GstVideoFrame to access its data pointers
    GstMapInfo mapInfo;
    gst_buffer_map(frame->buffer, &mapInfo, GST_MAP_WRITE);

    // Copy NV12 data to the GstVideoFrame
    guint8* data = mapInfo.data;

    // Copy Y plane data
    memcpy(data, nv12Image.data, nv12Image.rows * nv12Image.cols);

    // Copy UV plane data
    memcpy(data + nv12Image.rows * nv12Image.cols, nv12Image.data + nv12Image.rows * nv12Image.cols, nv12Image.rows * nv12Image.cols / 2);

    // Unmap the frame
    gst_buffer_unmap(frame->buffer, &mapInfo);
}

void filter(HailoROIPtr roi, GstVideoFrame *frame, gchar *current_stream_id)
// void filter(HailoROIPtr roi, cv::Mat& frame)
{
    // gint cv2_format = CV_8UC3;
    int font_thickness = 2;
    int line_thickness = 2;
    // guint matrix_width = (guint)GST_VIDEO_FRAME_WIDTH(frame);
    std::cout << "frame width: "<< frame->info.width << "height: " << frame->info.height << " format: " << frame->info.finfo->format << std::endl;

    // auto mat = cv::Mat(GST_VIDEO_FRAME_HEIGHT(frame), matrix_width, cv2_format,
    //                    GST_VIDEO_FRAME_PLANE_DATA(frame, 1), GST_VIDEO_FRAME_PLANE_STRIDE(frame, 1));
    // face_blur(frame, roi);
    GstVideoInfo *info = &frame->info;
    int width = GST_VIDEO_INFO_WIDTH(info);
    int height = GST_VIDEO_INFO_HEIGHT(info);
    GstVideoFormat format = GST_VIDEO_INFO_FORMAT(info);

    // Check if it's NV12 format
    if (format != GST_VIDEO_FORMAT_NV12) {
        throw std::runtime_error("Unsupported pixel format");
    }

    // Extract pointers and strides for the Y and UV planes
    guint8 *yData = (guint8*)GST_VIDEO_FRAME_PLANE_DATA(frame, 0); // Y plane
    int yStride = GST_VIDEO_FRAME_PLANE_STRIDE(frame, 0);
    guint8 *uvData = (guint8*)GST_VIDEO_FRAME_PLANE_DATA(frame, 1); // UV plane
    int uvStride = GST_VIDEO_FRAME_PLANE_STRIDE(frame, 1);

    // Create an OpenCV Mat for Y channel
    cv::Mat yMat(height, width, CV_8UC1, yData, yStride);

    // Create an OpenCV Mat for UV channels
    cv::Mat uvMat(height / 2, width / 2, CV_8UC2, uvData, uvStride);

    // Convert NV12 to BGR
    cv::Mat mat;
    cv::cvtColorTwoPlane(yMat, uvMat, mat, cv::COLOR_YUV2BGR_NV12);


    if (mat.channels() != 3) {
        throw std::runtime_error("Invalid number of channels in converted image");
    }

    for (auto obj : roi->get_objects())
    {
        switch (obj->get_type())
        {
        case HAILO_DETECTION:
        {
            HailoDetectionPtr detection = std::dynamic_pointer_cast<HailoDetection>(obj);
            draw_detection(mat, detection, roi, font_thickness, line_thickness);

            break;
        }
        default:
            // continue;
            std::cout << obj->get_type() << std::endl;
            break;
        }
    }

    mat = BGR2YUV_NV12(mat);

    // copyNv12DataToGstVideoFrame(mat, frame);

    mat.release();
}

void filter1(HailoROIPtr roi, GstVideoFrame *frame, gchar *current_stream_id)
// void filter(HailoROIPtr roi, cv::Mat& frame)
{
    // gint cv2_format = CV_8UC3;
    // int font_thickness = 2;
    // int line_thickness = 2;
    // // guint matrix_width = (guint)GST_VIDEO_FRAME_WIDTH(frame);
    // std::cout << "frame width: "<< frame->info.width << "height: " << frame->info.height << " format: " << frame->info.finfo->format << std::endl;

    // // auto mat = cv::Mat(GST_VIDEO_FRAME_HEIGHT(frame), matrix_width, cv2_format,
    // //                    GST_VIDEO_FRAME_PLANE_DATA(frame, 1), GST_VIDEO_FRAME_PLANE_STRIDE(frame, 1));
    // // face_blur(frame, roi);
    // GstVideoInfo *info = &frame->info;
    // int width = GST_VIDEO_INFO_WIDTH(info);
    // int height = GST_VIDEO_INFO_HEIGHT(info);
    // GstVideoFormat format = GST_VIDEO_INFO_FORMAT(info);

    // // Check if it's NV12 format
    // if (format != GST_VIDEO_FORMAT_NV12) {
    //     throw std::runtime_error("Unsupported pixel format");
    // }

    // // Extract pointers and strides for the Y and UV planes
    // guint8 *yData = (guint8*)GST_VIDEO_FRAME_PLANE_DATA(frame, 0); // Y plane
    // int yStride = GST_VIDEO_FRAME_PLANE_STRIDE(frame, 0);
    // guint8 *uvData = (guint8*)GST_VIDEO_FRAME_PLANE_DATA(frame, 1); // UV plane
    // int uvStride = GST_VIDEO_FRAME_PLANE_STRIDE(frame, 1);
    std::cout << "heloo" << std::endl;
    // // Create an OpenCV Mat for Y channel
    // std::cout << "before yMat" << std::endl;
    // cv::Mat yMat(height, width, CV_8UC1, yData, yStride);

    // // Create an OpenCV Mat for UV channels
    // std::cout << "before uvMat" << std::endl;
    // cv::Mat uvMat(height / 2, width / 2, CV_8UC2, uvData, uvStride);

    // std::cout << "after" << std::endl;
    // // Convert NV12 to BGR
    // cv::Mat mat;
    // cv::cvtColorTwoPlane(yMat, uvMat, mat, cv::COLOR_YUV2BGR_NV12);

    // std::cout << "-------" << mat.channels() << "-----------" << std::endl;

    // if (mat.channels() != 3) {
    //     throw std::runtime_error("Invalid number of channels in converted image");
    // }

    // printImageDimensions(mat);


    // for (auto obj : roi->get_objects())
    // {
    //     switch (obj->get_type())
    //     {
    //     case HAILO_DETECTION:
    //     {
    //         std::cout << "------------detection-----------------";
    //         HailoDetectionPtr detection = std::dynamic_pointer_cast<HailoDetection>(obj);
    //         // draw_detection(mat, detection, roi, font_thickness, line_thickness);
    //         auto global_id = get_global_id(detection);
    //         std::string id_text = std::to_string(global_id->get_id());
    //         std::cout << "Detect person id: " << id_text << std::endl;

    //         break;
    //     }
    //     default:
    //         // continue;
    //         std::cout << obj->get_type() << std::endl;
    //         break;
    //     }
    // }

    // mat.release();
}