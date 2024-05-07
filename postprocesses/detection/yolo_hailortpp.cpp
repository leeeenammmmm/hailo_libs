#include "hailo_nms_decode.hpp"
#include "yolo_hailortpp.hpp"
#include "common/labels/coco_eighty.hpp"
#include "common/labels/fire_smoke.hpp"
#include "common/labels/person_face.hpp"

#include <fstream>
#include <ctime>
#include <iomanip>


static const std::string DEFAULT_YOLOV5S_OUTPUT_LAYER = "yolov5s_nv12/yolov5_nms_postprocess";
static const std::string DEFAULT_YOLOV5M_OUTPUT_LAYER = "yolov5m_wo_spp_60p/yolov5_nms_postprocess";
static const std::string DEFAULT_YOLOV5M_VEHICLES_OUTPUT_LAYER = "yolov5m_vehicles/yolov5_nms_postprocess";
static const std::string DEFAULT_YOLOV8S_OUTPUT_LAYER = "yolov8s/yolov8_nms_postprocess";
static const std::string DEFAULT_YOLOV8M_OUTPUT_LAYER = "yolov8m/yolov8_nms_postprocess";

//--------------------------[-----[------------------]]
#define TUANIOT
#ifdef TUANIOT
const std::string data_roi = "data_ROI.txt";
const std::string data_fire_warning = "data.txt";
std::time_t last_write_time = 0;
//std::string last_label = "";

static const float native_width = 1920;
static const float native_height = 1080;
static float x_min = 0;
static float y_min = 0;
static float x_max = 1920;
static float y_max = 1080;

void read_txt(std::string file_name){
    std::ifstream check_file(file_name);
    if (!check_file.is_open()) {
        return;
    }
    std::ifstream file(file_name, std::ios::app);
    if (file.is_open()){
        std::string line;
        while (std::getline(file, line)) {
            // Process each line as needed
            std::stringstream ss(line);
            std::vector<float> numbers;
            float num;
            while (ss >> num) { // Read each number separated by space
                if(ss.fail()){
                    std::cout << "file data config has wrong format!" << std::endl;
                    break;
                }
                numbers.push_back(num); // Store the number in the vector
            }
            if(numbers.size() == 4){
                x_min = numbers[0];
                y_min = numbers[1];
                x_max = numbers[2];
                y_max = numbers[3];
                // std::cout << line << std::endl;
            }
            else{
                std::cout << "file data config has wrong format!" << std::endl;
            }
            
        }
        file.close();
    }
}

void write_txt(const std::string label, std::string file_name) {
    std::time_t now = std::time(nullptr);
    //if (label == last_label && std::difftime(now, last_write_time) < 5) {
    if (std::difftime(now, last_write_time) < 1) {
        return; 
    }

    std::tm* local_time = std::localtime(&now);
    char buffer[80];

    std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%S", local_time);
    // Lấy phần giây thập phân (microseconds)
    auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count() % 1000000;

    // In ra thời gian với định dạng mong muốn
    //std::cout << buffer << "." << std::setw(6) << std::setfill('0') << microseconds << "+07:00" << std::endl;
    
    std::ifstream check_file(file_name);
    if (!check_file.is_open()) {
        std::ofstream create_file(file_name);
        create_file.close();
    }

    std::ofstream file(file_name, std::ios::app);
    if (file.is_open()){
        file << label << " " << buffer << "." << std::setw(6) << std::setfill('0') << microseconds << "+00:00" << std::endl;
        file.close();

        last_write_time = now;
        //last_label = label;
    }

}


#endif



static std::map<uint8_t, std::string> yolo_vehicles_labels = {
    {0, "unlabeled"},
    {1, "car"}};

void yolov5(HailoROIPtr roi)
{
    if (!roi->has_tensors())
    {
        return;
    }
    auto post = HailoNMSDecode(roi->get_tensor(DEFAULT_YOLOV5M_OUTPUT_LAYER), common::coco_eighty);
    auto detections = post.decode<float32_t, common::hailo_bbox_float32_t>();
    hailo_common::add_detections(roi, detections);
}

void yolov5s_nv12(HailoROIPtr roi)
{
    if (!roi->has_tensors())
    {
        return;
    }
    auto post = HailoNMSDecode(roi->get_tensor(DEFAULT_YOLOV5S_OUTPUT_LAYER), common::coco_eighty);
    auto detections = post.decode<float32_t, common::hailo_bbox_float32_t>();
    hailo_common::add_detections(roi, detections);
}

void yolov8s(HailoROIPtr roi)
{
    
    if (!roi->has_tensors())
    {
        return;
    }
    auto post = HailoNMSDecode(roi->get_tensor(DEFAULT_YOLOV8S_OUTPUT_LAYER), common::coco_eighty);
    auto detections = post.decode<float32_t, common::hailo_bbox_float32_t>();
    hailo_common::add_detections(roi, detections);

}

void yolov8s_personface_in_ROI(HailoROIPtr roi)
{
    
    if (!roi->has_tensors())
    {
        return;
    }
    auto post = HailoNMSDecode(roi->get_tensor(DEFAULT_YOLOV8S_OUTPUT_LAYER), common::person_face);
    auto detections = post.decode<float32_t, common::hailo_bbox_float32_t>();
    //[feature] detect stranger in ROI
    std::vector<HailoDetection> stranger_detections;
    read_txt(data_roi);
    for(auto detection : detections){
        HailoBBox roi_bbox = hailo_common::create_flattened_bbox(roi->get_bbox(), roi->get_scaling_bbox());
        auto detection_bbox = detection.get_bbox();
        auto xmin = std::clamp<int>(((detection_bbox.xmin() * roi_bbox.width()) + roi_bbox.xmin()) * native_width, 0, native_width);
        auto ymin = std::clamp<int>(((detection_bbox.ymin() * roi_bbox.height()) + roi_bbox.ymin()) * native_height, 0, native_height);
        auto xmax = std::clamp<int>(((detection_bbox.xmax() * roi_bbox.width()) + roi_bbox.xmin()) * native_width, 0, native_width);
        auto ymax = std::clamp<int>(((detection_bbox.ymax() * roi_bbox.height()) + roi_bbox.ymin()) * native_height, 0, native_height);
        if(xmin < x_min || ymin < y_min || xmax > x_max || ymax > y_max)
            continue;
        if(detection.get_confidence() < 0.5)
            continue;
        stranger_detections.push_back(detection);
    }
    //-----end-----------
    hailo_common::add_detections(roi, stranger_detections);

}

void yolov8s_fire_smoke_warning(HailoROIPtr roi)
{
    
    if (!roi->has_tensors())
    {
        return;
    }
    auto post = HailoNMSDecode(roi->get_tensor(DEFAULT_YOLOV8S_OUTPUT_LAYER), common::fire_smoke);
    auto detections = post.decode<float32_t, common::hailo_bbox_float32_t>();
    //[feature] filter detection has confident >= 0.75
    std::vector<HailoDetection> high_confident_detections;
    for(auto detection : detections){
        if(detection.get_confidence() < 0.75)
            continue;
        high_confident_detections.push_back(detection);
    }
    //-----end-----------
    hailo_common::add_detections(roi, high_confident_detections);

    //[feature] warning fire smoke
    /*
        get detection from detections
        get label from detection
        use label to call function write txt 
    */
   for(auto detection : high_confident_detections){
        std::string label = detection.get_label();
        
        write_txt(label, data_fire_warning);

   }

}

void yolov8m(HailoROIPtr roi)
{
    if (!roi->has_tensors())
    {
        return;
    }
    auto post = HailoNMSDecode(roi->get_tensor(DEFAULT_YOLOV8M_OUTPUT_LAYER), common::coco_eighty);
    auto detections = post.decode<float32_t, common::hailo_bbox_float32_t>();
    hailo_common::add_detections(roi, detections);
}

void yolox(HailoROIPtr roi)
{
    auto post = HailoNMSDecode(roi->get_tensor("yolox_nms_postprocess"), common::coco_eighty);
    auto detections = post.decode<float32_t, common::hailo_bbox_float32_t>();
    hailo_common::add_detections(roi, detections);
}

void yolov5m_vehicles(HailoROIPtr roi)
{
    auto post = HailoNMSDecode(roi->get_tensor(DEFAULT_YOLOV5M_VEHICLES_OUTPUT_LAYER), yolo_vehicles_labels);
    auto detections = post.decode<float32_t, common::hailo_bbox_float32_t>();
    hailo_common::add_detections(roi, detections);
}

void yolov5_no_persons(HailoROIPtr roi)
{
    auto post = HailoNMSDecode(roi->get_tensor(DEFAULT_YOLOV5M_OUTPUT_LAYER), common::coco_eighty);
    auto detections = post.decode<float32_t, common::hailo_bbox_float32_t>();
    for (auto it = detections.begin(); it != detections.end();)
    {
        if (it->get_label() == "person")
        {
            it = detections.erase(it);
        }
        else
        {
            ++it;
        }
    }
    hailo_common::add_detections(roi, detections);
}

void filter(HailoROIPtr roi)
{
    yolov5(roi);
}