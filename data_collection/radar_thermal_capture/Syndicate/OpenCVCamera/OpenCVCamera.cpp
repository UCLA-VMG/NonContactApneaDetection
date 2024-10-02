#include "OpenCVCamera.h"

using namespace cv;

using namespace Syndicate;

OpenCVCamera::OpenCVCamera(ptree::value_type& sensor_settings, ptree::value_type& global_settings)
    : Syndicate::Camera(sensor_settings, global_settings),
    cameraID(sensor_settings.second.get<int>("camera_id"))
{
    if (sensor_settings.second.get<bool>("hardware_sync") == 0) {
        hardware_sync = false;
    }
    else {
        hardware_sync = true;
    }
    std::cout << std::endl << std::endl << "Configuring " << sensorName << "\n";
    //--- OPEN CAP
    cap = openCap(cameraID);
}

OpenCVCamera::~OpenCVCamera()
{
    // Deinitialize camera
    // the camera will be deinitialized automatically in VideoCapture destructor
}

void OpenCVCamera::AcquireSave(double seconds, boost::barrier& startBarrier)
{
    int num_frames(int(seconds)*int(fps));
    // int num_frames(int(seconds)*int(fps));
    std::cout << std::endl << std::endl << "*** IMAGE ACQUISITION ***" << std::endl << std::endl;
    startBarrier.wait();
    auto start = std::chrono::steady_clock::now();
    // if hardware sync is enabled, then use that signal to record all frames
    if (hardware_sync) {
        for (int i = 0; i < num_frames; i++)
        {
            cv::Mat frame;
            cv::Mat save_frame;
            // Wait for a new frame from camera and store it into 'frame'
            cap.read(frame);
            
            RecordTimeStamp();
            save_frame = std::any_cast<cv::Mat>(frame);
            // Create a unique filename
            std::ostringstream filename;
            filename << rootPath << sensorName << "_" << int(i) << ".tiff";
            std::vector<int> tags = {TIFFTAG_COMPRESSION, COMPRESSION_NONE};
            imwrite(filename.str().c_str(), save_frame,tags);

            logFile << sensorName << " " << std::to_string(int(i)) << std::endl;
        }
    }
    // if hardware sync is not enabled, then record at 30 fps by dropping every other frame
    // (since thermal camera natively records at 60 fps)
    else {
        // MULTIPLY NUM_FRAMES BY 2 BECAUSE REQUESTED FPS IS 30 BUT ACTUAL FPS IS 60
        for (int i = 0; i < int(num_frames*2); i++)
        {
            cv::Mat frame;
            cv::Mat save_frame;
            // Wait for a new frame from camera and store it into 'frame'
            cap.read(frame);
            
            if (i % 2 == 0) {
                RecordTimeStamp();
                save_frame = std::any_cast<cv::Mat>(frame);
                // Create a unique filename
                std::ostringstream filename;
                filename << rootPath << sensorName << "_" << int(i/2) << ".tiff";
                std::vector<int> tags = {TIFFTAG_COMPRESSION, COMPRESSION_NONE};
                imwrite(filename.str().c_str(), save_frame,tags);

                logFile << sensorName << " " << std::to_string(int(i/2)) << std::endl;
            }
        }
    }
    // end acquisition 
    auto end = std::chrono::steady_clock::now();
    std::cout << "Time Taken for " << sensorName  << " " << float((end-start).count())/1'000'000'000 << "\n";
    SaveTimeStamps();

    this->setHealthCode(HealthCode::ONLINE);
}

bool OpenCVCamera::setResolution(VideoCapture cap, double width = 640, double height = 512) {
    bool result = true;
    try {
        cap.set(CAP_PROP_FRAME_WIDTH, width);
        std::cout << "Width after using video.set(CAP_PROP_FRAME_WIDTH) : " << cap.get(CAP_PROP_FRAME_WIDTH) << std::endl;
        cap.set(CAP_PROP_FRAME_HEIGHT, height);
        std::cout << "Height after using video.set(CAP_PROP_FRAME_HEIGHT) : " << cap.get(CAP_PROP_FRAME_HEIGHT) << std::endl;
    }
    catch (Exception& e) {
        std::cout << "Error configuring resolution: " << e.what() << std::endl;
        result = false;
    }
    return result;
}

bool OpenCVCamera::setFps(VideoCapture cap, double fps) {
    bool result = true;
    try {
        cap.set(CAP_PROP_FPS, fps);
        std::cout << "Frames per second using video.get(CAP_PROP_FPS) : " << fps << std::endl;
    }
    catch (Exception& e) {
        std::cout << "Error configuring fps: " << e.what() << std::endl;
        result = false;
    }
    return result;
}

bool OpenCVCamera::configure(cv::VideoCapture cap, int cameraID, double fps = 30.0, double height = 512, double width = 640) {
    bool result = true;
    try {
        // Print the device serial number
        std::cout << "Configuring device " << cameraID << std::endl;

        // set fps
        if (!setFps(cap, fps)) {
            return false;
        }
        // set resolution
        if (!setResolution(cap, height, width)) {
            std::cout << "Couldn't set resolution to height " << height << " and width " << width << std::endl;
            return false;
        }
    }
    catch (Exception& e) {
        std::cout << "Error configuring camera: " << e.what() << std::endl;
        result = false;
    }
    return result;
}

bool OpenCVCamera::getResolution(VideoCapture cap) {
    bool result = true;
    try {
        double width = cap.get(CAP_PROP_FRAME_WIDTH);
        std::cout << "Width using video.get(CAP_PROP_FRAME_WIDTH) : " << width << std::endl;
        double height = cap.get(CAP_PROP_FRAME_HEIGHT);
        std::cout << "Height using video.get(CAP_PROP_FRAME_HEIGHT) : " << height << std::endl;
    }
    catch (Exception& e) {
        std::cout << "Error configuring resolution: " << e.what() << std::endl;
        result = false;
    }
    return result;
}

bool OpenCVCamera::getFps(VideoCapture cap) {
    bool result = true;
    try {
        double fps = cap.get(CAP_PROP_FPS);
        std::cout << "Frames per second using video.get(CAP_PROP_FPS) : " << fps << std::endl;
    }
    catch (Exception& e) {
        std::cout << "Error configuring fps: " << e.what() << std::endl;
        result = false;
    }
    return result;
}

VideoCapture OpenCVCamera::openCap(int cameraID) {
    //--- INITIALIZE CAP
    VideoCapture cap;
    // open the default camera using default API
    // cap.open(0);
    // // OR advance usage: select any API backend 
    int apiID = cv::CAP_DSHOW;      // 0 = autodetect default API
                                  // open selected camera using selected API
    cap.open(cameraID + apiID);
    // check if we succeeded
    if (!cap.isOpened()) {
        std::cerr << "ERROR! Unable to open camera\n";
    }

    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    std::cout << "Width after using video.set(CAP_PROP_FRAME_WIDTH) : " << cap.get(CAP_PROP_FRAME_WIDTH) << std::endl;
    
    cap.set(CAP_PROP_FRAME_HEIGHT, 512);
    std::cout << "Height after using video.set(CAP_PROP_FRAME_HEIGHT) : " << cap.get(CAP_PROP_FRAME_HEIGHT) << std::endl;

    // // Doesn't work because boson does not support setting FPS via OpenCV, and by default is set to 60 fps
    // // therefore to get 30 fps, simply discard every other frame @ 60 fps
    // cap.set(CAP_PROP_FPS, 30 );
    // std::cout << "FPS after using video.set(CAP_PROP_FPS) : " << cap.get(CAP_PROP_FPS) << std::endl;

    cap.set(CAP_PROP_CONVERT_RGB, FALSE);
    std::cout << "CAP_PROP_CONVERT_RGB : " << cap.get(CAP_PROP_CONVERT_RGB) << std::endl;

    cap.set(CAP_PROP_FOURCC, cv::VideoWriter::fourcc('Y','1','6',' '));
    int fourcc = cap.get(CAP_PROP_FOURCC);
    std::string fourcc_str = format("%c%c%c%c", fourcc & 255, (fourcc >> 8) & 255, (fourcc >> 16) & 255, (fourcc >> 24) & 255);
    std::cout << "CAP_PROP_FOURCC: " << fourcc_str << std::endl;

    return cap;
}

const std::string currentDateTime() {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
    // for more information about date/time format
    strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);

    return buf;
}
