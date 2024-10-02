#pragma once
#include <WinSock2.h>
#include "camera.h"
#include "Spinnaker.h"
#include "SpinGenApi/SpinnakerGenApi.h"

#include <chrono>
#include <thread>
#include <string>
#include <memory>
#include <any>
#include <iostream>
#include <iomanip>
#include <map>

struct SpinnakerCamera : public Syndicate::Camera 
{
    SpinnakerCamera(ptree::value_type& sensor_settings, ptree::value_type& global_settings);
    ~SpinnakerCamera();

    void AcquireSave(double seconds, boost::barrier& startBarrier);

    Spinnaker::SystemPtr system;
    Spinnaker::CameraList camList;
    Spinnaker::CameraPtr flir_cam;
    
    std::string cameraID;
    std::string cameraName;

    std::string pixelFormat;
    int offset_x;
    int offset_y;
    
    int bin_size;
    double exposure_compensation;
    double exposure_time;
    double gain;
    double black_level;
};

// using namespace Spinnaker;
// using namespace Spinnaker::GenApi;
// using namespace Spinnaker::GenICam;

Spinnaker::CameraPtr getCam(Spinnaker::SystemPtr system, Spinnaker::CameraList camList, std::string serialNumber);

bool configure(Spinnaker::CameraPtr pCam, Spinnaker::GenApi::INodeMap& nodeMap, std::string cameraName, double fps, int width, int height, int offset_x, int offset_y, std::string pixelFormat, int bin_size, double exposure_compensation, double exposure_time, double gain, double black_level, bool hardwareSync, bool primary);

std::string GetDeviceSerial(Spinnaker::CameraPtr pCam);

bool setPrimary(Spinnaker::GenApi::INodeMap& nodeMap, std::string& cameraName);

bool setSecondary(Spinnaker::GenApi::INodeMap& nodeMap);
