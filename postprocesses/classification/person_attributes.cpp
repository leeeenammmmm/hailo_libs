/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/
#include <vector>
#include "common/labels/peta.hpp"
#include "common/tensors.hpp"
#include "common/math.hpp"
#include "hailo_tracker.hpp"
#include "person_attributes.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xarray.hpp"

//[nam] include libpqxx
#include <pqxx/pqxx> 
#include <ctime>
#include <chrono>

#define RESNET_V1_18_PERSON_OUTPUT_LAYER_NAME "person_attr_resnet_v1_18/fc1"
#define RESNET_V1_18_PERSON_OUTPUT_LAYER_NAME_NV12 "person_attr_resnet_v1_18_nv12/fc1"
#define RESNET_V1_18_PERSON_THRESHOLD 0.7f

std::string tracker_name = "hailo_person_tracker";

xt::xarray<float> get_attr_predictions_from_tensor(HailoTensorPtr outp_tensor)
{
    // Convert the tensor to xarray
    xt::xarray<float> xscores = common::get_xtensor_float(outp_tensor);
    auto attr_predictions = xt::view(xscores, 0, 0, xt::all());

    // Calculate the face attributes values by sigmoid
    common::sigmoid(attr_predictions.data(), attr_predictions.size());
    return attr_predictions;
}

std::string current_timestamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm now_tm = *std::localtime(&now_time);

    std::ostringstream oss;
    oss << std::put_time(&now_tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

void connect_to_db(int tracking_id, int Age_Young, int Age_Adult, int Age_Old, int Gender_Male, int Gender_Female, int Hair_Length_Short, int Hair_Length_Long, int Hair_Length_Bald, int UpperBody_Length_Short, int UpperBody_Length_Long, int UpperBody_Color_Black, int UpperBody_Color_Blue, int UpperBody_Color_Brown, int UpperBody_Color_Green, int UpperBody_Color_Grey, int UpperBody_Color_Orange, int UpperBody_Color_Pink, int UpperBody_Color_Purple, int UpperBody_Color_Red, int UpperBody_Color_White, int UpperBody_Color_Yellow, int UpperBody_Color_Other, int LowerBody_Length_Short, int LowerBody_Length_Long, int LowerBody_Color_Black, int LowerBody_Color_Blue, int LowerBody_Color_Brown, int LowerBody_Color_Green, int LowerBody_Color_Grey, int LowerBody_Color_Orange, int LowerBody_Color_Pink, int LowerBody_Color_Purple, int LowerBody_Color_Red, int LowerBody_Color_White, int LowerBody_Color_Yellow, int LowerBody_Color_Other, int LowerBody_Type_Trousers_And_Shorts, int LowerBody_Type_Skirt_And_Dress, int Accessory_Backpack, int Accessory_NoBackpack, int Accessory_Bag, int Accessory_NoBag, int Accessory_Glasses_Normal, int Accessory_Glasses_Sun, int Accessory_NoGlasses, int Accessory_Hat, int Accessory_NoHat){
    try {
        pqxx::connection C("dbname = testdb user = postgres password = postgres \
        hostaddr = 127.0.0.1 port = 5432");
        if (C.is_open()) {
            std::cout << "Opened database successfully: " << C.dbname() << std::endl;

        } else {
            std::cout << "Can't open database" << std::endl;
            return;
        }

        // Check if the tracking_id already exists
        pqxx::work W(C);
        std::string check_sql = "SELECT COUNT(*) FROM person_tracking WHERE tracking_id = " + W.quote(tracking_id) + ";";
        pqxx::result R = W.exec(check_sql);
        // Fetch the count from the result
        int count = R[0][0].as<int>();
        if (count > 0){
            std::cout << "tracking_id " << tracking_id << " already exists." << std::endl;
            std::string event_timestamp = current_timestamp();
            std::string update_sql = "UPDATE person_tracking SET age_young = " + W.quote(Age_Young) + ", age_adult = " + W.quote(Age_Adult) + ", age_old = " + W.quote(Age_Old) + ", gender_male = " + W.quote(Gender_Male) + ", gender_female = " + W.quote(Gender_Female) + ", hair_length_short = " + W.quote(Hair_Length_Short) + ", hair_length_long = " + W.quote(Hair_Length_Long) + ", hair_length_bald = " + W.quote(Hair_Length_Bald) + ", upperbody_length_short = " + W.quote(UpperBody_Length_Short) + ", upperbody_length_long = " + W.quote(UpperBody_Length_Long) + ", UpperBody_Color_Black = " + W.quote(UpperBody_Color_Black) + ", UpperBody_Color_Blue = " + W.quote(UpperBody_Color_Blue) + ", UpperBody_Color_Brown = " + W.quote(UpperBody_Color_Brown) + ", UpperBody_Color_Green = " + W.quote(UpperBody_Color_Green) + ", UpperBody_Color_Grey = " + W.quote(UpperBody_Color_Grey) + ", UpperBody_Color_Orange = " + W.quote(UpperBody_Color_Orange) + ", UpperBody_Color_Pink = " + W.quote(UpperBody_Color_Pink) + ", UpperBody_Color_Purple = " + W.quote(UpperBody_Color_Purple) + ", UpperBody_Color_Red = " + W.quote(UpperBody_Color_Red) + ", UpperBody_Color_White = " + W.quote(UpperBody_Color_White) + ", UpperBody_Color_Yellow = " + W.quote(UpperBody_Color_Yellow) + ", UpperBody_Color_Other = " + W.quote(UpperBody_Color_Other) + ", LowerBody_Length_Short = " + W.quote(LowerBody_Length_Short) + ", LowerBody_Length_Long = " + W.quote(LowerBody_Length_Long) + ", LowerBody_Color_Black = " + W.quote(LowerBody_Color_Black) + ", LowerBody_Color_Blue = " + W.quote(LowerBody_Color_Blue) + ", LowerBody_Color_Brown = " + W.quote(LowerBody_Color_Brown) + ", LowerBody_Color_Green = " + W.quote(LowerBody_Color_Green) + ", LowerBody_Color_Grey = " + W.quote(LowerBody_Color_Grey) + ", LowerBody_Color_Orange = " + W.quote(LowerBody_Color_Orange) + ", LowerBody_Color_Pink = " + W.quote(LowerBody_Color_Pink) + ", LowerBody_Color_Purple = " + W.quote(LowerBody_Color_Purple) + ", LowerBody_Color_Red = " + W.quote(LowerBody_Color_Red) + ", LowerBody_Color_White = " + W.quote(LowerBody_Color_White) + ", LowerBody_Color_Yellow = " + W.quote(LowerBody_Color_Yellow) + ", LowerBody_Color_Other = " + W.quote(LowerBody_Color_Other) + ", LowerBody_Type_Trousers_And_Shorts = " + W.quote(LowerBody_Type_Trousers_And_Shorts) + ", LowerBody_Type_Skirt_And_Dress = " + W.quote(LowerBody_Type_Skirt_And_Dress) + ", Accessory_Backpack = " + W.quote(Accessory_Backpack) + ", Accessory_NoBackpack = " + W.quote(Accessory_NoBackpack) + ", Accessory_Bag = " + W.quote(Accessory_Bag) + ", Accessory_NoBag = " + W.quote(Accessory_NoBag) + ", Accessory_Glasses_Normal = " + W.quote(Accessory_Glasses_Normal) + ", Accessory_Glasses_Sun = " + W.quote(Accessory_Glasses_Sun) + ", Accessory_NoGlasses = " + W.quote(Accessory_NoGlasses) + ", Accessory_Hat = " + W.quote(Accessory_Hat) + ", Accessory_NoHat = " + W.quote(Accessory_NoHat) + ", end_time = " + W.quote(event_timestamp) + " WHERE tracking_id = " + W.quote(tracking_id) + ";";
            W.exec(update_sql);
            W.commit();
            return;
        }

        

        /* Create SQL statement */
        std::string event_timestamp = current_timestamp();
        std::string sql = "INSERT INTO person_tracking (tracking_id,Age_Young,Age_Adult,Age_Old,Gender_Male,Gender_Female,Hair_Length_Short,Hair_Length_Long,Hair_Length_Bald,UpperBody_Length_Short,UpperBody_Length_Long,UpperBody_Color_Black,UpperBody_Color_Blue,UpperBody_Color_Brown,UpperBody_Color_Green,UpperBody_Color_Grey,UpperBody_Color_Orange,UpperBody_Color_Pink,UpperBody_Color_Purple,UpperBody_Color_Red,UpperBody_Color_White,UpperBody_Color_Yellow,UpperBody_Color_Other,LowerBody_Length_Short,LowerBody_Length_Long,LowerBody_Color_Black,LowerBody_Color_Blue,LowerBody_Color_Brown,LowerBody_Color_Green,LowerBody_Color_Grey,LowerBody_Color_Orange,LowerBody_Color_Pink,LowerBody_Color_Purple,LowerBody_Color_Red,LowerBody_Color_White,LowerBody_Color_Yellow,LowerBody_Color_Other,LowerBody_Type_Trousers_And_Shorts,LowerBody_Type_Skirt_And_Dress,Accessory_Backpack,Accessory_NoBackpack,Accessory_Bag,Accessory_NoBag,Accessory_Glasses_Normal,Accessory_Glasses_Sun,Accessory_NoGlasses,Accessory_Hat,Accessory_NoHat,start_time) "  \
            "VALUES (" + W.quote(tracking_id) + "," + W.quote(Age_Young) + ","  + W.quote(Age_Adult) + "," + W.quote(Age_Old) + "," + W.quote(Gender_Male) + "," + W.quote(Gender_Female) + "," + W.quote(Hair_Length_Short) + "," + W.quote(Hair_Length_Long) + "," + W.quote(Hair_Length_Bald) + "," + W.quote(UpperBody_Length_Short) + "," + W.quote(UpperBody_Length_Long) + "," + W.quote(UpperBody_Color_Black) + "," + W.quote(UpperBody_Color_Blue) + "," + W.quote(UpperBody_Color_Brown) + "," + W.quote(UpperBody_Color_Green) + "," + W.quote(UpperBody_Color_Grey) + "," + W.quote(UpperBody_Color_Orange) + "," + W.quote(UpperBody_Color_Pink) + "," + W.quote(UpperBody_Color_Purple) + "," + W.quote(UpperBody_Color_Red) + "," + W.quote(UpperBody_Color_White) + "," + W.quote(UpperBody_Color_Yellow) + "," + W.quote(UpperBody_Color_Other) + "," + W.quote(LowerBody_Length_Short) + "," + W.quote(LowerBody_Length_Long) + "," + W.quote(LowerBody_Color_Black) + "," + W.quote(LowerBody_Color_Blue) + "," + W.quote(LowerBody_Color_Brown) + "," + W.quote(LowerBody_Color_Green) + "," + W.quote(LowerBody_Color_Grey) + "," + W.quote(LowerBody_Color_Orange) + "," + W.quote(LowerBody_Color_Pink) + "," + W.quote(LowerBody_Color_Purple) + "," + W.quote(LowerBody_Color_Red) + "," + W.quote(LowerBody_Color_White) + "," + W.quote(LowerBody_Color_Yellow) + "," + W.quote(LowerBody_Color_Other) + "," + W.quote(LowerBody_Type_Trousers_And_Shorts) + "," + W.quote(LowerBody_Type_Skirt_And_Dress) + "," + W.quote(Accessory_Backpack) + "," + W.quote(Accessory_NoBackpack) + "," + W.quote(Accessory_Bag) + "," + W.quote(Accessory_NoBag) + "," + W.quote(Accessory_Glasses_Normal) + "," + W.quote(Accessory_Glasses_Sun) + "," + W.quote(Accessory_NoGlasses) + "," + W.quote(Accessory_Hat) + "," + W.quote(Accessory_NoHat) + "," + W.quote(event_timestamp) + ");";

        // Execute the SQL query
        W.exec(sql);

        // Commit the transaction
        W.commit();


        C.disconnect ();
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return;
    }
}

void person_attributes_postprocess(HailoROIPtr roi, std::string output_layer_name)
{
    if (!roi->has_tensors())
    {
        return;
    }

    // Extract the relevant output tensor.
    HailoTensorPtr outp_tensor = roi->get_tensor(output_layer_name);
    auto attr_predictions = get_attr_predictions_from_tensor(outp_tensor);

    std::string label = "";
    std::string jde_tracker_name = tracker_name + "_" + roi->get_stream_id();
    auto unique_ids = hailo_common::get_hailo_unique_id(roi);
    if (unique_ids.size() == 1)
    {
        HailoTracker::GetInstance().remove_classifications_from_track(jde_tracker_name,
                                                                      unique_ids[0]->get_id(),
                                                                      std::string("person_attributes"));
    }

    uint num_of_attributes = attr_predictions.shape()[0];
    // Iterate over the attribute predictions
    for (uint i = 0; i < num_of_attributes; i++)
    {
        // Get the confidence
        float confidence = attr_predictions(i);
        // Get the label from the peta labels
        label = labels::peta_filtered[i];

        // Filter confidence values by threshold
        HailoClassificationPtr classification;
        if (label != "" && confidence > RESNET_V1_18_PERSON_THRESHOLD)
        {
            classification = std::make_shared<HailoClassification>(std::string("person_attributes"),
                                                                   i,
                                                                   label,
                                                                   confidence);
        }
        else if(label == "Male")
        {
            classification = std::make_shared<HailoClassification>(std::string("person_attributes"),
                                                        i,
                                                        "Female",
                                                        confidence);
        }

        if (!classification)
            continue;

        if (unique_ids.empty())
        {
            hailo_common::add_object(roi, classification);
        }
        else if(unique_ids.size() == 1)
        {
            // We are updating the tracker with the results.
            // No need to add the object to the ROI because it is followed by fakesing - end of sub-pipeline.
            HailoTracker::GetInstance().add_object_to_track(jde_tracker_name,
                                                            unique_ids[0]->get_id(),
                                                            classification);
        }
    }
}

void person_attributes_postprocess_47(HailoROIPtr roi, std::string output_layer_name)
{
    if (!roi->has_tensors())
    {
        return;
    }

    //[NAM] reset attr in roi
    std::vector<HailoObjectPtr> objects;
    for(auto o : roi->get_objects()){
        if(o->get_type() == 1){
            objects.push_back(o);
        }
    }
    hailo_common::remove_objects(roi, objects);

    // Extract the relevant output tensor.
    HailoTensorPtr outp_tensor = roi->get_tensor(output_layer_name);
    auto attr_predictions = get_attr_predictions_from_tensor(outp_tensor);

    std::string label = "";
    std::string jde_tracker_name = tracker_name + "_" + roi->get_stream_id();
    auto unique_ids = hailo_common::get_hailo_unique_id(roi);
    if (unique_ids.size() == 1)
    {
        HailoTracker::GetInstance().remove_classifications_from_track(jde_tracker_name,
                                                                      unique_ids[0]->get_id(),
                                                                      std::string("person_attributes"));
    }

    uint num_of_attributes = attr_predictions.shape()[0];
    //[nam] declare a vector store attr
    std::vector<int> attr;
    // Iterate over the attribute predictions
    for (uint i = 0; i < num_of_attributes; i++)
    {
        // Get the confidence
        float confidence = attr_predictions(i);
        //[nam] if confidence > threshold, push back 1 to vector attr
        if(confidence > RESNET_V1_18_PERSON_THRESHOLD){
            attr.push_back(1);
        }
        else
            attr.push_back(0);
        // Get the label from the peta labels
        label = labels::person_attr_filter[i];

        // Filter confidence values by threshold
        HailoClassificationPtr classification;
        if (label != "" && confidence > RESNET_V1_18_PERSON_THRESHOLD)
        {
            classification = std::make_shared<HailoClassification>(std::string("person_attributes"),
                                                                   i,
                                                                   label,
                                                                   confidence);
            // std::cout << label << " " << confidence << std::endl;
        }

        if (!classification)
            continue;

        if (unique_ids.empty())
        {
            hailo_common::add_object(roi, classification);
        }
        else if(unique_ids.size() == 1)
        {
            // We are updating the tracker with the results.
            // No need to add the object to the ROI because it is followed by fakesing - end of sub-pipeline.
            // HailoTracker::GetInstance().add_object_to_track(jde_tracker_name,
            //                                                 unique_ids[0]->get_id(),
            //                                                 classification);
            hailo_common::add_object(roi, classification);
        }
    }
    if(unique_ids.size() == 1){
        //[nam] connect to db
        connect_to_db(unique_ids[0]->get_id(), attr[0], attr[1], attr[2], attr[3], attr[4], attr[5], attr[6], attr[7], attr[8], attr[9], attr[10], attr[11], attr[12], attr[13], attr[14], attr[15], attr[16], attr[17], attr[18], attr[19], attr[20], attr[21], attr[22], attr[23], attr[24], attr[25], attr[26], attr[27], attr[28], attr[29], attr[30], attr[31], attr[32], attr[33], attr[34], attr[35], attr[36], attr[37], attr[38], attr[39], attr[40], attr[41], attr[42], attr[43], attr[44], attr[45], attr[46]);
    }
}

void filter(HailoROIPtr roi)
{
    person_attributes_postprocess(roi, RESNET_V1_18_PERSON_OUTPUT_LAYER_NAME);
}

void person_attributes_nv12(HailoROIPtr roi)
{
    person_attributes_postprocess(roi, RESNET_V1_18_PERSON_OUTPUT_LAYER_NAME_NV12);
}

void person_attributes_rgba(HailoROIPtr roi)
{
    person_attributes_postprocess(roi, "person_attr_resnet_v1_18_rgbx/fc1");
}

void filter_47_classes(HailoROIPtr roi)
{
    person_attributes_postprocess_47(roi, RESNET_V1_18_PERSON_OUTPUT_LAYER_NAME);
}

void filter_nv12_47_classes(HailoROIPtr roi)
{
    person_attributes_postprocess_47(roi, RESNET_V1_18_PERSON_OUTPUT_LAYER_NAME_NV12);
}
