#include "camera.h"

namespace Syndicate
{
    Camera::Camera(ptree::value_type& sensor_settings, ptree::value_type& global_settings)
        : Sensor(sensor_settings, global_settings),  fps(sensor_settings.second.get<int>("fps")),
        height(sensor_settings.second.get<int>("height")),
        width(sensor_settings.second.get<int>("width"))

    {
        if(sensor_settings.second.find("type") != sensor_settings.second.not_found())
        {
            cameraType = std::any_cast<std::string>(sensor_settings.second.get<std::string>("type"));
        }
    }

    
    Camera::~Camera()
    {
    }
}