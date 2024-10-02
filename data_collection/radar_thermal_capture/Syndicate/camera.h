#pragma once
#include <WinSock2.h>
#include "sensor.h"

#include <chrono>
#include <thread>
#include <string>
#include <memory>
#include <any>

namespace Syndicate
{
    struct Camera : public Sensor 
    {
        Camera(ptree::value_type& sensor_settings, ptree::value_type& global_settings);
        ~Camera();

        virtual void AcquireSave(double seconds, boost::barrier& startBarrier) = 0;

        // Operating Characteristics
        std::string cameraType;
        double fps;
        double height;
        double width;
    };
}