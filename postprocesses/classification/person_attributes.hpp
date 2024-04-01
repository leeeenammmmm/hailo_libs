/**
* Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
* Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
**/
#pragma once
#include "hailo_objects.hpp"
#include "hailo_common.hpp"

__BEGIN_DECLS
void filter(HailoROIPtr roi);
void filter_47_classes(HailoROIPtr roi);
void filter_nv12_47_classes(HailoROIPtr roi);
void person_attributes_nv12(HailoROIPtr roi);
void person_attributes_rgba(HailoROIPtr roi);
__END_DECLS