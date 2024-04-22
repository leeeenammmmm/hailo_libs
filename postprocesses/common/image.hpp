/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/
/**
 * @file overlay/common.hpp
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2022-01-20
 *
 * @copyright Copyright (c) 2022
 *
 */

#pragma once

#include <gst/video/video.h>
#include "hailomat.hpp"
#include "hailo_objects.hpp"

__BEGIN_DECLS

/**
 * @brief Get the size of buffer with specific caps
 *
 * @param caps The caps to extract size from.
 * @return size_t The size of a buffer with those caps.
 */
size_t get_size(GstCaps *caps);

/**
 * @brief Get the mat object
 *
 * @param info The GstVideoInfo to extract mat from.
 * @param data Pointer to the data.
 * @return cv::Mat The mat object.
 */
cv::Mat get_mat_from_video_info(GstVideoInfo *info, char *data);

/**
 * @brief Get the mat object
 *
 * @param info The GstVideoInfo to extract mat from.
 * @param map The GstMapInfo to extract mat from.
 * @return cv::Mat The mat object.
 */
cv::Mat get_mat(GstVideoInfo *info, GstMapInfo *map);

/**
 * @brief Get the mat object from given GstVideoFrame
 *
 * @param info The GstVideoFrame to extract mat from.
 * @return cv::Mat The mat object.
 */
cv::Mat get_mat_from_gst_frame(GstVideoFrame *frame);

/**
 * @brief Resizes a YUY2 image (4 channel cv::Mat)
 *
 * @param cropped_image - cv::Mat &
 *        The cropped image to resize
 *
 * @param resized_image - cv::Mat &
 *        The resized image container to fill
 *        (dims for resizing are assumed from here)
 *
 * @param interpolation - int
 *        The interpolation type to resize by.
 *        Must be a supported opencv type
 *        (bilinear, nearest neighbors, etc...)
 */
void resize_yuy2(cv::Mat &cropped_image, cv::Mat &resized_image, int interpolation = cv::INTER_LINEAR);

/**
 * @brief Resizes a NV12 image (1 channel cv::Mat)
 *
 * @param cropped_image - cv::Mat &
 *        The cropped image to resize
 *
 * @param resized_image - cv::Mat &
 *        The resized image container to fill
 *        (dims for resizing are assumed from here)
 *
 * @param interpolation - int
 *        The interpolation type to resize by.
 *        Must be a supported opencv type
 *        (bilinear, nearest neighbors, etc...)
 */
void resize_nv12(std::vector<cv::Mat> &cropped_image_vec, std::vector<cv::Mat> &resized_image_vec, int interpolation = cv::INTER_LINEAR);

/**
 * @brief Resize an image using Letterbox strategy
 *
 * @param cropped_image - cv::Mat &
 *        The cropped image to resize
 *
 * @param resized_image - cv::Mat &
 *        The resized image container to fill
 *        (dims for resizing are assumed from here)
 *
 * @param color - cv::Scalar
 *        The color to fill the letterbox with
 *
 * @param interpolation - int
 *        The interpolation type to resize by.
 *        Must be a supported opencv type
 *        (bilinear, nearest neighbors, etc...)
 */
HailoBBox resize_letterbox_rgb(cv::Mat &cropped_image, cv::Mat &resized_image, cv::Scalar color, int interpolation = cv::INTER_LINEAR);

/**
 * @brief Resize an NV12 image using Letterbox strategy
 *
 * @param cropped_image - cv::Mat &
 *        The cropped image to resize
 *
 * @param resized_image - cv::Mat &
 *        The resized image container to fill
 *        (dims for resizing are assumed from here)
 *
 * @param color - cv::Scalar
 *        The color to fill the letterbox with
 *
 * @param interpolation - int
 *        The interpolation type to resize by.
 *        Must be a supported opencv type
 *        (bilinear, nearest neighbors, etc...)
 */
HailoBBox resize_letterbox_nv12(std::vector<cv::Mat> &cropped_image_vec, std::vector<cv::Mat> &resized_image_vec, cv::Scalar color, int interpolation = cv::INTER_LINEAR);
__END_DECLS

std::shared_ptr<HailoMat> get_mat_by_format_buffer(GstBuffer *buffer, GstVideoInfo *info, int line_thickness=1, int font_thickness=1)
{
    std::shared_ptr<HailoMat> hmat = nullptr;
    GstVideoFrame frame;
#ifdef IMX6_TARGET
    bool success = gst_video_frame_map(&frame, info, buffer, GstMapFlags(GST_MAP_READ | GST_MAP_WRITE));
#else
    bool success = gst_video_frame_map(&frame, info, buffer, GstMapFlags(GST_MAP_READ));
#endif

    if (!success)
    {
        gst_video_frame_unmap(&frame);
        GST_CAT_ERROR(GST_CAT_DEFAULT, "Failed to map buffer to video frame, Buffer may be not writable");
        throw std::runtime_error("Failed to map buffer to video frame, Buffer may be not writable");
    }

    uint8_t *plane0_data = (uint8_t *)GST_VIDEO_FRAME_PLANE_DATA(&frame, 0);

    switch (GST_VIDEO_INFO_FORMAT(info))
    {
    case GST_VIDEO_FORMAT_RGB:
    {
        hmat = std::make_shared<HailoRGBMat>(plane0_data,
                                             GST_VIDEO_INFO_HEIGHT(info),
                                             GST_VIDEO_INFO_WIDTH(info),
                                             GST_VIDEO_INFO_PLANE_STRIDE(info, 0),
                                             line_thickness,
                                             font_thickness);
        break;
    }
    case GST_VIDEO_FORMAT_RGBA:
    {
        hmat = std::make_shared<HailoRGBAMat>(plane0_data,
                                              GST_VIDEO_INFO_HEIGHT(info),
                                              GST_VIDEO_INFO_WIDTH(info),
                                              GST_VIDEO_INFO_PLANE_STRIDE(info, 0),
                                              line_thickness,
                                              font_thickness);
        break;
    }
    case GST_VIDEO_FORMAT_YUY2:
    {
        hmat = std::make_shared<HailoYUY2Mat>(plane0_data,
                                              GST_VIDEO_INFO_HEIGHT(info),
                                              GST_VIDEO_INFO_WIDTH(info),
                                              GST_VIDEO_INFO_PLANE_STRIDE(info, 0),
                                              line_thickness,
                                              font_thickness);
        break;
    }
    case GST_VIDEO_FORMAT_NV12:
    {
        hmat = std::make_shared<HailoNV12Mat>(plane0_data,
                                              GST_VIDEO_INFO_HEIGHT(info),
                                              GST_VIDEO_INFO_WIDTH(info),
                                              GST_VIDEO_INFO_PLANE_STRIDE(info, 0),
                                              GST_VIDEO_INFO_PLANE_STRIDE(info, 1),
                                              line_thickness,
                                              font_thickness,
                                              plane0_data,
                                              GST_VIDEO_FRAME_PLANE_DATA(&frame, 1));
        break;
    }

    default:
        break;
    }

    gst_video_frame_unmap(&frame);
    return hmat;
}

std::shared_ptr<HailoMat> get_mat_by_format(GstVideoFrame *frame, int line_thickness=1, int font_thickness=1)
{
    std::shared_ptr<HailoMat> hmat = nullptr;
    GstVideoInfo *info = &frame->info;

    uint8_t *plane0_data = (uint8_t *)GST_VIDEO_FRAME_PLANE_DATA(frame, 0);

    switch (GST_VIDEO_INFO_FORMAT(info))
    {
    case GST_VIDEO_FORMAT_RGB:
    {
        hmat = std::make_shared<HailoRGBMat>(plane0_data,
                                             GST_VIDEO_INFO_HEIGHT(info),
                                             GST_VIDEO_INFO_WIDTH(info),
                                             GST_VIDEO_INFO_PLANE_STRIDE(info, 0),
                                             line_thickness,
                                             font_thickness);
        break;
    }
    case GST_VIDEO_FORMAT_RGBA:
    {
        hmat = std::make_shared<HailoRGBAMat>(plane0_data,
                                              GST_VIDEO_INFO_HEIGHT(info),
                                              GST_VIDEO_INFO_WIDTH(info),
                                              GST_VIDEO_INFO_PLANE_STRIDE(info, 0),
                                              line_thickness,
                                              font_thickness);
        break;
    }
    case GST_VIDEO_FORMAT_YUY2:
    {
        hmat = std::make_shared<HailoYUY2Mat>(plane0_data,
                                              GST_VIDEO_INFO_HEIGHT(info),
                                              GST_VIDEO_INFO_WIDTH(info),
                                              GST_VIDEO_INFO_PLANE_STRIDE(info, 0),
                                              line_thickness,
                                              font_thickness);
        break;
    }
    case GST_VIDEO_FORMAT_NV12:
    {
        hmat = std::make_shared<HailoNV12Mat>(plane0_data,
                                              GST_VIDEO_INFO_HEIGHT(info),
                                              GST_VIDEO_INFO_WIDTH(info),
                                              GST_VIDEO_INFO_PLANE_STRIDE(info, 0),
                                              GST_VIDEO_INFO_PLANE_STRIDE(info, 1),
                                              line_thickness,
                                              font_thickness,
                                              plane0_data,
                                              GST_VIDEO_FRAME_PLANE_DATA(frame, 1));
        break;
    }

    default:
        break;
    }

    return hmat;
}
