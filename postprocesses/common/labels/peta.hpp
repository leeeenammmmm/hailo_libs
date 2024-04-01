/**
* Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
* Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
**/

#pragma once
#include <map>
#include <stdint.h>
#include <string>

namespace labels
{
    static std::map<uint8_t, std::string> person_attr = {
        {0, "Age-Young"},
        {1, "Age-Adult"},
        {2, "Age-Old"},
        {3, "Gender-Male"},
        {4, "Gender-Female"},
        {5, "Hair-Length-Short"},
        {6, "Hair-Length-Long"},
        {7, "Hair-Length-Bald"},
        {8, "UpperBody-Length-Short"},
        {9, "UpperBody-Length-Long"},
        {10, "UpperBody-Color-Black"},
        {11, "UpperBody-Color-Blue"},
        {12, "UpperBody-Color-Brown"},
        {13, "UpperBody-Color-Green"},
        {14, "UpperBody-Color-Grey"},
        {15, "UpperBody-Color-Orange"},
        {16, "UpperBody-Color-Pink"},
        {17, "UpperBody-Color-Purple"},
        {18, "UpperBody-Color-Red"},
        {19, "UpperBody-Color-White"},
        {20, "UpperBody-Color-Yellow"},
        {21, "UpperBody-Color-Other"},
        {22, "LowerBody-Length-Short"},
        {23, "LowerBody-Length-Long"},
        {24, "LowerBody-Color-Black"},
        {25, "LowerBody-Color-Blue"},
        {26, "LowerBody-Color-Brown"},
        {27, "LowerBody-Color-Green"},
        {28, "LowerBody-Color-Grey"},
        {29, "LowerBody-Color-Orange"},
        {30, "LowerBody-Color-Pink"},
        {31, "LowerBody-Color-Purple"},
        {32, "LowerBody-Color-Red"},
        {33, "LowerBody-Color-White"},
        {34, "LowerBody-Color-Yellow"},
        {35, "LowerBody-Color-Other"},
        {36, "LowerBody-Type-Trousers&Shorts"},
        {37, "LowerBody-Type-Skirt&Dress"},
        {38, "Accessory-Backpack"},
        {39, "Accessory-NoBackpack"},
        {40, "Accessory-Bag"},
        {41, "Accessory-NoBag"},
        {42, "Accessory-Glasses-Normal"},
        {43, "Accessory-Glasses-Sun"},
        {44, "Accessory-NoGlasses"},
        {45, "Accessory-Hat"},
        {46, "Accessory-NoHat"}};
    static std::map<uint8_t, std::string> peta = {
        {0, "Age16-30"},
        {1, "Age31-45"},
        {2, "Age46-60"},
        {3, "AgeAbove61"},
        {4, "Backpack"},
        {5, "CarryingOther"},
        {6, "Casual lower"},
        {7, "Casual upper"},
        {8, "Formal lower"},
        {9, "Formal upper"},
        {10, "Hat"},
        {11, "Jacket"},
        {12, "Jeans"},
        {13, "Leather shoes"},
        {14, "Logo"},
        {15, "Long hair"},
        {16, "Male"},
        {17, "Messenger bag"},
        {18, "Muffler"},
        {19, "No accesory"},
        {20, "No carrying"},
        {21, "Plaid"},
        {22, "Plastic bag"},
        {23, "Sandals"}, 
        {24, "Shoes"},
        {25, "Shorts"},
        {26, "Short sleeve"},
        {27, "Skirt"},
        {28, "Sneaker"},
        {29, "Stripes"},
        {30, "Sunglasses"},
        {31, "Trousers"},
        {32, "T-shirt"},
        {33, "UpperOther"},
        {34, "V-Neck"}};

    static std::map<uint8_t, std::string> peta_filtered = {
        {0, "Age < 30"},
        {1, "Age 31-45"},
        {2, "Age 46-60"},
        {3, "Age 60+"},
        {4, ""},
        {5, ""},
        {6, ""},
        {7, ""},
        {8, ""},
        {9, ""},
        {10, "Hat"},
        {11, ""},
        {12, ""},
        {13, ""},
        {14, "Logo"},
        {15, "Long hair"},
        {16, "Male"},
        {17, ""},
        {18, "Muffler"},
        {19, ""},
        {20, ""},
        {21, ""},
        {22, "Plastic bag"},
        {23, ""}, 
        {24, ""},
        {25, ""},
        {26, ""},
        {27, ""},
        {28, ""},
        {29, ""},
        {30, "Sunglasses"},
        {31, ""},
        {32, ""},
        {33, ""},
        {34, ""}};        
}