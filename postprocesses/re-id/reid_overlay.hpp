/**
* Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
* Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
**/
#pragma once
#include <opencv2/opencv.hpp>
#include "hailo_objects.hpp"
#include <gst/video/video-format.h>
#include <gst/gst.h>
#include <gst/gstbuffer.h>
#include <gst/video/video.h>
#include "common/hailomat.hpp"
#include "common/image.hpp"

__BEGIN_DECLS

void filter(HailoROIPtr roi, GstVideoFrame *frame, char *current_stream_id);
// void filter(HailoROIPtr roi, cv::Mat& frame);
// HailoUniqueIDPtr get_global_id(HailoDetectionPtr detection);
cv::Scalar indexToColor(size_t index);

__END_DECLS