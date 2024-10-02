#pragma once

#include "camera.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <windows.h>
#include <ctime>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <tiff.h>
#include <tiffio.h>

#include <chrono>
#include <thread>
#include <string>
#include <memory>
#include <any>
#include <iostream>
#include <iomanip>
#include <map>

struct OpenCVCamera : public Syndicate::Camera 
{
    OpenCVCamera(ptree::value_type& sensor_settings, ptree::value_type& global_settings);
    ~OpenCVCamera();

    void AcquireSave(double seconds, boost::barrier& startBarrier);

    bool setResolution(cv::VideoCapture cap, double width, double height);
    bool setFps(cv::VideoCapture cap, double fps);
    bool AcquireImages(cv::VideoCapture cap, int num_frames);
    bool getResolution(cv::VideoCapture cap);
    bool getFps(cv::VideoCapture cap);

    cv::VideoCapture openCap(int cameraID);
    bool configure(cv::VideoCapture cap, int deviceID, double fps, double height, double width);

    int cameraID;
    cv::VideoCapture cap;
    bool hardware_sync;

};

const std::string currentDateTime();
