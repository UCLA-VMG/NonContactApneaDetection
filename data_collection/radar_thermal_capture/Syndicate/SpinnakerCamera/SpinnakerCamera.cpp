#include "SpinnakerCamera.h"

using namespace Spinnaker;
using namespace Spinnaker::GenApi;
using namespace Spinnaker::GenICam;
using namespace std;

using namespace Syndicate;

SpinnakerCamera::SpinnakerCamera(ptree::value_type& sensor_settings, ptree::value_type& global_settings)
    : Syndicate::Camera(sensor_settings, global_settings), 
    cameraID(sensor_settings.second.get<std::string>("camera_id")),
    cameraName(sensor_settings.second.get<std::string>("camera_name")),
    pixelFormat(sensor_settings.second.get<std::string>("pixel_format")),
    offset_x(sensor_settings.second.get<int>("offset_x")),
    offset_y(sensor_settings.second.get<int>("offset_y")),
    bin_size(sensor_settings.second.get<int>("binning_size")),
    exposure_compensation(sensor_settings.second.get<double>("exposure_compensation")),
    exposure_time(sensor_settings.second.get<double>("exposure_time")),
    gain(sensor_settings.second.get<double>("gain")),
    black_level(sensor_settings.second.get<double>("black_level"))
{
    
    std::cout << "  Original FPS: " << fps << endl;
    std::cout << "  Original Width: " << width << endl;
    std::cout << "  Original Height: " << height << endl;
    std::cout << "  Original OffsetX: " << offset_x << endl;
    std::cout << "  Original OffsetY: " << offset_y << endl;
    std::cout << "  Original Pixel Format: " << pixelFormat << endl;
    std::cout << "  Original Binning Size: " << bin_size << endl;
    std::cout << "  Original Exposure Compensation: " << exposure_compensation << endl;
    std::cout << "  Original Exposure Time: " << exposure_time << endl;
    std::cout << "  Original Gain: " << gain << endl;
    std::cout << "  Original Black Level: " << black_level << endl;
    
    // Retrieve singleton reference to system object
    system = System::GetInstance();
    // Retrieve list of cameras from the system
    camList = system->GetCameras();
    // Retrieve camera.
    flir_cam = getCam(system, camList, cameraID);

    // Initialize camera
    flir_cam->Init();
    // Get NodeMap and nodeMapTLDevice
    INodeMap& nodeMap = flir_cam->GetNodeMap();
    INodeMap& nodeMapTLDevice = flir_cam->GetTLDeviceNodeMap();

    // Configure camera with specified fps/height/width
    try {
        std::cout << endl << endl << "Configuring " << sensorName << "\n";
        if (!configure(flir_cam, nodeMap, cameraName, fps, width, height, offset_x, offset_y, pixelFormat, bin_size, exposure_compensation, exposure_time, gain, black_level, hardwareSync, primary)) {
            std::cout << "Camera configuration for device " << cameraID << " unsuccessful, aborting...";
        }
    }
    catch (Spinnaker::Exception& e) {
        std::cout << "Error: " << e.what() << "\n";
    }
    // Enable Hardware Synchronization if asked for
    if(hardwareSync)
    {
        if(primary)
        {
            setPrimary(nodeMap, cameraName);
        }
        if(!primary)
        {
            setSecondary(nodeMap);
        }
    }

    this->setHealthCode(HealthCode::ONLINE);
}

SpinnakerCamera::~SpinnakerCamera()
{
    // Deinitialize camera
    flir_cam->DeInit();
}

void SpinnakerCamera::AcquireSave(double seconds, boost::barrier& startBarrier)
{
    // 0. Initialize Starting Variables
    this->setHealthCode(HealthCode::RUNNING);
    int result = 0;
    int num_frames(seconds*fps);
    ImagePtr convertedImage = nullptr;

    // 1. Wait Until all other sensors have reached here
    startBarrier.wait();

    auto start = std::chrono::steady_clock::now();
    // 2.0 Begin acquiring images
    flir_cam->BeginAcquisition();
    const unsigned int k_numImages = num_frames;

    for (unsigned int imageCnt = 0; imageCnt < k_numImages; imageCnt++) 
    {
        try 
        {
            // 2.1 Retrieve next received image
            ImagePtr pResultImage = flir_cam->GetNextImage(1000);
            // 2.2 Save Software System Timestamp
            RecordTimeStamp();
            // 2.3 Ensure image completion:
            if (pResultImage->IsIncomplete()) 
            {
                // Retrieve and print the image status description
                logFile << "Image incomplete: " 
                        << Image::GetImageStatusDescription(pResultImage->GetImageStatus())
                        << "..." << endl
                        << endl;
            }
            // 2.4 If Image is Complete:
            else 
            {
                // std::cout << "Hello!" << std::endl;
                if(pixelFormat == "Mono"){
                    convertedImage = pResultImage->Convert(PixelFormat_Mono8, NO_COLOR_PROCESSING);
                }
                else if(pixelFormat == "RGB"){
                    convertedImage = pResultImage->Convert(PixelFormat_BGR8, NO_COLOR_PROCESSING);
                }
                else if(pixelFormat == "pRGB"){
                    convertedImage = pResultImage->Convert(PixelFormat_BayerRGPolarized8, NO_COLOR_PROCESSING);
                }
                else
                    std::cout << "Error: Pixel Format Invalid. \n";
                // runningBuffer.push(convertedImage);
                ostringstream filename;
                filename << rootPath << sensorName << "_" << imageCnt;
                convertedImage->Save(filename.str().c_str(),  Spinnaker::ImageFileFormat::BMP);
                logFile << sensorName << " " << pResultImage->GetFrameID() << std::endl;
                // // Retrieve timestamp
                // ChunkData chunkData = pResultImage->GetChunkData();
                // uint64_t timestamp = chunkData.GetTimestamp();
                // // cout << "\tTimestamp: " << timestamp << endl;
            }
            // Release image
            pResultImage->Release();
            // std::cout << endl;
        }
        catch (Spinnaker::Exception& e) {
            std::cout << "Error: " << e.what() << endl;
            result = -1;
        }
    }
    // End acquisition
    flir_cam->EndAcquisition();
    auto end = std::chrono::steady_clock::now();
    std::cout <<"Time Taken for acquistion " << sensorName << " " << static_cast<double>((end-start).count())/1'000'000'000 << "\n";
    logFile << "Time Taken for acquistion " << sensorName << " " << static_cast<double>((end-start).count())/1'000'000'000 << "\n";
    SaveTimeStamps();

    this->setHealthCode(HealthCode::ONLINE);
}

// This helper function retrieves the device serial number from the cameras nodemap.
std::string GetDeviceSerial(Spinnaker::CameraPtr pCam) {
    INodeMap& nodeMap = pCam->GetTLDeviceNodeMap();
    CStringPtr ptrDeviceSerialNumber = nodeMap.GetNode("DeviceSerialNumber");
    if ((IsAvailable(ptrDeviceSerialNumber)) && (IsReadable(ptrDeviceSerialNumber))) {
        std::cout << string(ptrDeviceSerialNumber->GetValue()) << endl;
        return string(ptrDeviceSerialNumber->GetValue());
    }
    return "";
}

bool setPrimary(INodeMap& nodeMap, std::string& cameraName)
{
    if(cameraName == "BackFlyS") {
        std::cout << "***********\n";
        CEnumerationPtr ptrLine = nodeMap.GetNode("LineSelector");
        CEnumEntryPtr ptrLine1 = ptrLine->GetEntryByName("Line1");
        ptrLine->SetIntValue(ptrLine1->GetValue());

        CEnumerationPtr ptrLineMode = nodeMap.GetNode("LineMode");
        CEnumEntryPtr ptrLineModeVal = ptrLineMode->GetEntryByName("Output");
        ptrLineMode->SetIntValue(ptrLineModeVal->GetValue());

        CEnumEntryPtr ptrLine2 = ptrLine->GetEntryByName("Line2");
        ptrLine->SetIntValue(ptrLine2->GetValue());

        CBooleanPtr ptr3_3V = nodeMap.GetNode("V3_3Enable");
        ptr3_3V->SetValue(true);
        std::cout << "****done*******\n";
        return true;
    }
    else if (cameraName == "Grasshopper3") {
        std::cout << "***********\n";
        CEnumerationPtr ptrLine = nodeMap.GetNode("LineSelector");
        CEnumEntryPtr ptrLine1 = ptrLine->GetEntryByName("Line2");
        ptrLine->SetIntValue(ptrLine1->GetValue());

        CEnumerationPtr ptrLineMode = nodeMap.GetNode("LineMode");
        CEnumEntryPtr ptrLineModeVal = ptrLineMode->GetEntryByName("Output");
        ptrLineMode->SetIntValue(ptrLineModeVal->GetValue());

        CEnumerationPtr ptrLineSource = nodeMap.GetNode("LineSource");
        CEnumEntryPtr ptrLineSourceVal = ptrLineSource->GetEntryByName("ExposureActive");
        ptrLineSource->SetIntValue(ptrLineSourceVal->GetValue());
        return true;
    }
    else
    {
        return false;
    }
    return false;
}

bool setSecondary(INodeMap& nodeMap)
{
    std::cout << "Setting sec 1\n";
    CEnumerationPtr ptrTriggerMode = nodeMap.GetNode("TriggerMode");
    CEnumEntryPtr ptrTriggerModeOff = ptrTriggerMode->GetEntryByName("Off");
    ptrTriggerMode->SetIntValue(ptrTriggerModeOff->GetValue());

    CEnumerationPtr ptrTriggerSelector = nodeMap.GetNode("TriggerSelector");
    CEnumEntryPtr ptrTriggerSelectorFrameStart = ptrTriggerSelector->GetEntryByName("FrameStart");
    ptrTriggerSelector->SetIntValue(ptrTriggerSelectorFrameStart->GetValue());

    CEnumerationPtr ptrTriggerSource = nodeMap.GetNode("TriggerSource");
    CEnumEntryPtr ptrTriggerSourceHardware = ptrTriggerSource->GetEntryByName("Line3");
    ptrTriggerSource->SetIntValue(ptrTriggerSourceHardware->GetValue());

    CEnumEntryPtr ptrTriggerModeOn = ptrTriggerMode->GetEntryByName("On");
    ptrTriggerMode->SetIntValue(ptrTriggerModeOn->GetValue());

    CEnumerationPtr ptrTriggerOverlap = nodeMap.GetNode("TriggerOverlap");
    CEnumEntryPtr ptrTriggerOverlapReadOut = ptrTriggerOverlap->GetEntryByName("ReadOut");
    ptrTriggerOverlap->SetIntValue(ptrTriggerOverlapReadOut->GetValue());
    std::cout << "Setting sec 2\n";
    return true;
}

bool setResolution(INodeMap& nodeMap, int width, int height, int offset_x, int offset_y) {
    bool result = true;
    try {        
        // Set width
        CIntegerPtr ptrWidth = nodeMap.GetNode("Width");
        if(!IsWritable(ptrWidth->GetAccessMode()) || !IsReadable(ptrWidth->GetAccessMode()) || ptrWidth->GetInc() == 0 || ptrWidth->GetMax() == 0) {
            std::cout << "Unable to read or write to Width. Aborting..." << endl;
            return false;
        }
        ptrWidth->SetValue(width);
        int widthToSet = static_cast<int>(ptrWidth->GetValue());
        std::cout <<    "Width: " << widthToSet << endl;

        // Set height
        CIntegerPtr ptrHeight = nodeMap.GetNode("Height");
        if(!IsWritable(ptrHeight->GetAccessMode()) || !IsReadable(ptrHeight->GetAccessMode()) || ptrHeight->GetInc() == 0 || ptrHeight->GetMax() == 0) {
            std::cout <<    "Unable to read or write to Height. Aborting..." << endl;
            return false;
        }
        ptrHeight->SetValue(height);
        int heightToSet = static_cast<int>(ptrHeight->GetValue());
        std::cout <<    "Height: " << heightToSet << endl;
        // CIntegerPtr ptrWidth = nodeMap.GetNode("Width");

        // Get width Max
        int widthMax = static_cast<int>(ptrWidth->GetMax());
        std::cout <<    "Width Max: " << widthMax << endl;

        // Get width Max
        int heightMax = static_cast<int>(ptrHeight->GetMax());
        std::cout <<    "Height Max: " << heightMax << endl;

        // Set offset_x
        if (width != widthMax) {
            CIntegerPtr ptrOffsetX = nodeMap.GetNode("OffsetX");
            if(!IsWritable(ptrOffsetX->GetAccessMode()) || !IsReadable(ptrOffsetX->GetAccessMode()) || ptrOffsetX->GetInc() == 0 || ptrOffsetX->GetMax() == 0) {
                std::cout <<    "Unable to read or write to OffsetX. Aborting..." << endl;
                return false;
            }
            ptrOffsetX->SetValue(offset_x);
            int offsetXToSet = static_cast<int>(ptrOffsetX->GetValue());
            std::cout <<    "offset_x: " << offsetXToSet << endl;
        }

        if (height != heightMax) { 
            // Set offset_y
            CIntegerPtr ptrOffsetY = nodeMap.GetNode("OffsetY");
            if(!IsWritable(ptrOffsetY->GetAccessMode()) || !IsReadable(ptrOffsetY->GetAccessMode()) || ptrOffsetY->GetInc() == 0 || ptrOffsetY->GetMax() == 0) {
                std::cout <<    "Unable to read or write to OffsetY. Aborting..." << endl;
                return false;
            }
            ptrOffsetY->SetValue(offset_y);
            int offsetYToSet = static_cast<int>(ptrOffsetY->GetValue());
            std::cout <<    "offset_y: " << offsetYToSet << endl;
        }

    }
    catch (Exception& e) {
        std::cout <<    "Error Configuring resolution: " << e.what() << endl;
        result = false;
    }
    return result;
}

bool setBinning(INodeMap& nodeMap, int bin_size) {
    bool result = true;
    try {
        // Set height_bin
        CIntegerPtr ptrHeightBin = nodeMap.GetNode("BinningVertical");
        if(!IsWritable(ptrHeightBin->GetAccessMode()) || !IsReadable(ptrHeightBin->GetAccessMode()) || ptrHeightBin->GetInc() == 0 || ptrHeightBin->GetMax() == 0) {
            std::cout << "Unable to read or write to Binning Size. Aborting..." << endl;
            return false;
        }
        ptrHeightBin->SetValue(bin_size);
        int heightBinToSet = static_cast<int>(ptrHeightBin->GetValue());
        std::cout <<    "Bin Size: " << heightBinToSet << endl;
    }
    catch (Exception& e) {
        std::cout << "Error Configuring Bining: " << e.what() << endl;
        result = false;
    }
    return result;
}

bool setFps(INodeMap& nodeMap, float fps, std::string cameraName) {
    // Set Default Result Value = True
    bool result = true;
    // Get Ptr to Node Acquisition Frame Rate to Check Prev Value
    CFloatPtr ptrAcquisitionFrameRate = nodeMap.GetNode("AcquisitionFrameRate");
    double frameRateToSet = static_cast<double>(ptrAcquisitionFrameRate->GetValue());
    cout << "   Prev FPS: " << frameRateToSet << endl;
    // Try Setting Acquisition Frame Rate
    try {
        if (cameraName == "Grasshopper3") {
            try {
                // Set AcquisitionFrameRateAuto to false
                CEnumerationPtr ptrAcquisitionFrameRateAuto = nodeMap.GetNode("AcquisitionFrameRateAuto");
                CEnumEntryPtr node_frame_rate_auto_off = ptrAcquisitionFrameRateAuto->GetEntryByName("Off");
                const int64_t frame_rate_auto_off = node_frame_rate_auto_off->GetValue();
                ptrAcquisitionFrameRateAuto->SetIntValue(frame_rate_auto_off);
            }
            catch (Exception& e) {std::cout << "Unable to Change Acquisition Frame Rate Auto to Off. \n";}
        }
        
        // Set AcquisitionFrameRateEnable to true
        CBooleanPtr ptrAcquisitionFrameRateEnable;
        if (cameraName == "Grasshopper3") {
            ptrAcquisitionFrameRateEnable = nodeMap.GetNode("AcquisitionFrameRateEnabled");
            try {
                ptrAcquisitionFrameRateEnable->SetValue(true);
            }
            catch(Exception& e) {
                cout << "Unable to set AcquisitionFrameRateEnabled. Aborting..." << endl;
                std::cerr << e.what() << '\n'; // "Could not enable frame rate: {0}".format(ex)
            }
        }
        else if (cameraName == "BlackFlyS") {
            ptrAcquisitionFrameRateEnable = nodeMap.GetNode("AcquisitionFrameRateEnable");
            try {
                ptrAcquisitionFrameRateEnable->SetValue(true);
            }
            catch(Exception& e) {
                cout << "Unable to set AcquisitionFrameRateEnabled. Aborting..." << endl;
                std::cerr << e.what() << '\n'; // "Could not enable frame rate: {0}".format(ex)
            }
        }
        else {
            cout << "cameraName not recognized. Unable to acess AcquisitionFrameRateEnable. Aborting..." << endl;
            return false;
        }

        // If AcquisitionFrameRate is Readable and Writable ...
        CFloatPtr ptrAcquisitionFrameRate = nodeMap.GetNode("AcquisitionFrameRate");
        if(!IsWritable(ptrAcquisitionFrameRate->GetAccessMode()) || !IsReadable(ptrAcquisitionFrameRate->GetAccessMode()) ) {
            cout << "Unable to set FPS. Aborting..." << endl;
            return false;
        }
        // ... Set FPS
        ptrAcquisitionFrameRate->SetValue(fps);
        double frameRateToSet = static_cast<double>(ptrAcquisitionFrameRate->GetValue());
        // cout << "FPS to be set to " << frameRateToSet << "..." << endl;
    }
    catch (Exception& e) {
        cout << "Error Configuring FPS: " << e.what() << endl;
        result = false;
    }
    // Get Ptr to Node Acquisition Frame Rate to Check New Value
    ptrAcquisitionFrameRate = nodeMap.GetNode("AcquisitionFrameRate");
    frameRateToSet = static_cast<double>(ptrAcquisitionFrameRate->GetValue());
    cout << "   New FPS: " << frameRateToSet << endl;

    return result;
}


bool setContinuousAcquisitionMode(INodeMap& nodeMap) {
    bool result = true;
    try {
        // Set acquisition mode to continuous
        CEnumerationPtr ptrAcquisitionMode = nodeMap.GetNode("AcquisitionMode");
        if (!IsWritable(ptrAcquisitionMode)) {
            std::cout << "Unable to set acquisition mode to continuous (node retrieval). Aborting..." << endl;
            return false;
        }
        CEnumEntryPtr ptrAcquisitionModeContinuous = ptrAcquisitionMode->GetEntryByName("Continuous");
        if (!IsReadable(ptrAcquisitionModeContinuous)) {
            std::cout << "Unable to set acquisition mode to continuous (enum entry retrieval). Aborting..." << endl;
            return false;
        }
        const int64_t acquisitionModeContinuous = ptrAcquisitionModeContinuous->GetValue();
        ptrAcquisitionMode->SetIntValue(acquisitionModeContinuous);
        std::cout << "ptrAcquisitionMode: "<< ptrAcquisitionMode->GetEntryByName("Continuous") << endl;
    }
    catch (Exception& e) {
        std::cout << "Error Configuring Acquisition Mode: " << e.what() << endl;
        result = false;
    }
    return result;
}

bool setPixelFormat(INodeMap& nodeMap, std::string pixelFormat) {
    bool result = true;
    try {
        CEnumerationPtr ptrPixelFormat = nodeMap.GetNode("PixelFormat");
        if (IsAvailable(ptrPixelFormat) && IsWritable(ptrPixelFormat))
        {
            // Retrieve the desired entry node from the enumeration node
            CEnumEntryPtr ptrPixelFormatMono8 = ptrPixelFormat->GetEntryByName("BayerRG8");
            if(pixelFormat == "Mono"){
                ptrPixelFormatMono8 = ptrPixelFormat->GetEntryByName("Mono8");
            }
            else if(pixelFormat == "pRGB"){
                ptrPixelFormatMono8 = ptrPixelFormat->GetEntryByName("BayerRGPolarized8");
            }
            else if(pixelFormat == "RGB"){
                ptrPixelFormatMono8 = ptrPixelFormat->GetEntryByName("BayerRG8");
            }
            else {
                cout << "Pixel format set to " << ptrPixelFormat->GetCurrentEntry()->GetSymbolic() << "..." << endl;
            }

            if (IsAvailable(ptrPixelFormatMono8) && IsReadable(ptrPixelFormatMono8))
            {
                // Retrieve the integer value from the entry node
                int64_t pixelFormatMono8 = ptrPixelFormatMono8->GetValue();
                // Set integer as new value for enumeration node
                ptrPixelFormat->SetIntValue(pixelFormatMono8);
                cout << "Pixel format set to " << ptrPixelFormat->GetCurrentEntry()->GetSymbolic() << "..." << endl;
            }
            else
            {
                cout << "Pixel format not available..." << endl;
            }
        }
        else
        {
            cout << "Pixel format not available..." << endl;
        }
    }
    catch (Exception& e) {
        std::cout << "Error Configuring Acquisition Mode: " << e.what() << endl;
        result = false;
    }
    return result;
}

bool setExposureCompensation(INodeMap& nodeMap, double exposure_compensation) {
    // Set Default Result Value = True
    bool result = true;
    // Get Ptr to Node Exposure Compensation to Check Prev Value
    CFloatPtr ptrExposureCompensation = nodeMap.GetNode("pgrExposureCompensation");
    double exposureCompensationToSet = static_cast<double>(ptrExposureCompensation->GetValue());
    std::cout << "  Prev Exposure Compensation: " << exposureCompensationToSet << endl;
    // Try Setting Exposure Compensation
    try {
        // Try Setting Exposure Compensation Auto to Off
        try {
            CEnumerationPtr ptrExposureCompensationAuto = nodeMap.GetNode("pgrExposureCompensationAuto");
            CEnumEntryPtr ptrExposureCompensationAutoOff = ptrExposureCompensationAuto->GetEntryByName("Off");
            const int64_t exposureCompensationAuto = ptrExposureCompensationAutoOff->GetValue();
            ptrExposureCompensationAuto->SetIntValue(exposureCompensationAuto);
        }
        catch (Exception& e) {std::cout << "Unable to change ptrExposureCompensationAuto to Off. \n";}
        // If ExposureCompensation is Readable and Writable ...
        CFloatPtr ptrExposureCompensation = nodeMap.GetNode("pgrExposureCompensation");
        if(!IsWritable(ptrExposureCompensation->GetAccessMode()) || !IsReadable(ptrExposureCompensation->GetAccessMode()) ) {
            std::cout << "Unable to set Exposure Compensation. Aborting..." << endl;
            return false;
        }
        // ... Set Exposure Compensation
        ptrExposureCompensation->SetValue(exposure_compensation);
        double exposureCompensationToSet = static_cast<double>(ptrExposureCompensation->GetValue());
        // std::cout << "Exposure Compensation to be set to " << exposureCompensationToSet << "..." << endl;
    }
    catch (Exception& e) {
        std::cout << "Error Configuring Exposure Compensation: " << e.what() << endl;
        result = false;
    }
    // Get Ptr to Node Exposure Compensation to Check New Value
    ptrExposureCompensation = nodeMap.GetNode("pgrExposureCompensation");
    exposureCompensationToSet = static_cast<double>(ptrExposureCompensation->GetValue());
    std::cout << "  New Exposure Compensation: " << exposureCompensationToSet << endl;
    return result;
}

bool setExposureTime(INodeMap& nodeMap, double exposure_time) {
    // Set Default Result Value = True
    bool result = true;
    // Get Ptr to Node Exposure Time to Check Prev Value
    CFloatPtr ptrExposureTime = nodeMap.GetNode("ExposureTime");
    double exposureTimeToSet = static_cast<double>(ptrExposureTime->GetValue());
    std::cout << "  Prev Exposure Time: " << exposureTimeToSet << endl;
    // Try Setting Exposure Time
    try {
        // Try Setting Exposure Time Auto to Off
        try {
            CEnumerationPtr ptrExposureMode = nodeMap.GetNode("ExposureMode");
            CEnumEntryPtr ptrExposureModeTimed = ptrExposureMode->GetEntryByName("Timed");
            // ptrExposureMode->SetIntValue(ptrExposureModeTimed->GetValue());
            ptrExposureMode->SetIntValue(static_cast<int64_t>(ptrExposureModeTimed->GetValue()));
        }
        catch (Exception& e) {std::cout << "Unable to change ptrExposureMode to Timed. \n";}
        // Try Setting Exposure Time Auto to Off
        try {
            CEnumerationPtr ptrExposureAuto = nodeMap.GetNode("ExposureAuto");
            CEnumEntryPtr ptrExposureAutoOff = ptrExposureAuto->GetEntryByName("Off");
            // ptrExposureAuto->SetIntValue(ptrExposureAutoOff->GetValue());
            ptrExposureAuto->SetIntValue(static_cast<int64_t>(ptrExposureAutoOff->GetValue()));
        }
        catch (Exception& e) {std::cout << "Unable to change ptrExposureAuto to Off. \n";}
        // If ExposureTime is Readable and Writable ..
        CFloatPtr ptrExposureTime = nodeMap.GetNode("ExposureTime");
        if(!IsWritable(ptrExposureTime->GetAccessMode()) || !IsReadable(ptrExposureTime->GetAccessMode()) ) {
            std::cout << "Unable to set Exposure Time. Aborting..." << endl;
            return false;
        }
        // ... Set Exposure Time
        ptrExposureTime->SetValue(exposure_time);
        double exposureTimeToSet = static_cast<double>(ptrExposureTime->GetValue());
        // std::cout << "Exposure Time to be set to " << exposureTimeToSet << "..." << endl;
    }
    catch (Exception& e) {
        std::cout << "Error Configuring Exposure Time: " << e.what() << endl;
        result = false;
    }
    // Get Ptr to Node Exposure Time to Check New Value
    ptrExposureTime = nodeMap.GetNode("ExposureTime");
    exposureTimeToSet = static_cast<double>(ptrExposureTime->GetValue());
    std::cout << "  New Exposure Time: " << exposureTimeToSet << endl;

    return result;
}

bool setGain(INodeMap& nodeMap, double gain) {
    // Set Default Result Value = True
    bool result = true;
    // Get Ptr to Node Gain to Check Prev Value
    CFloatPtr ptrGain = nodeMap.GetNode("Gain");
    double gainToSet = static_cast<double>(ptrGain->GetValue());
    std::cout << "  Prev Gain: " << gainToSet << endl;
    // Try Setting Gain
    try {
        try {
            // Set GainAuto to false
            CEnumerationPtr ptrGainAuto = nodeMap.GetNode("GainAuto");
            CEnumEntryPtr ptrGainAutoOff = ptrGainAuto->GetEntryByName("Off");
            ptrGainAuto->SetIntValue(static_cast<int64_t>(ptrGainAutoOff->GetValue()));
        }
        catch (Exception& e) {std::cout << "Unable to change GainAuto to Off. \n";}
        // If Gain is Readable and Writable ..
        CFloatPtr ptrGain = nodeMap.GetNode("Gain");
        if(!IsWritable(ptrGain->GetAccessMode()) || !IsReadable(ptrGain->GetAccessMode()) ) {
            cout << "Unable to set Gain. Aborting..." << endl;
            return false;
        }
        // ... Set Gain
        ptrGain->SetValue(gain);
        double gainToSet = static_cast<double>(ptrGain->GetValue());
        // cout << "Gain to be set to " << gainToSet << "..." << endl;
    }
    catch (Exception& e) {
        cout << "Error Configuring Gain: " << e.what() << endl;
        result = false;
    }
    // Get Ptr to Node Gain to Check New Value
    ptrGain = nodeMap.GetNode("Gain");
    gainToSet = static_cast<double>(ptrGain->GetValue());
    cout << "  New Gain: " << gainToSet << endl;

    return result;
}

bool setBlackLevel(INodeMap& nodeMap, double black_level) {
    // Set Default Result Value = True
    bool result = true;
    // Get Ptr to Node Gain to Check Prev Value
    CFloatPtr ptrBlackLevel = nodeMap.GetNode("BlackLevel");
    double blackLevelToSet = static_cast<double>(ptrBlackLevel->GetValue());
    std::cout << "  Prev Black Level: " << blackLevelToSet << endl;
    // Try Setting Black Level
    try {
        // If Black Level is Readable and Writable ..
        CFloatPtr ptrBlackLevel = nodeMap.GetNode("BlackLevel");
        if(!IsWritable(ptrBlackLevel->GetAccessMode()) || !IsReadable(ptrBlackLevel->GetAccessMode()) ) {
            cout << "Unable to set Black Level. Aborting..." << endl;
            return false;
        }
        // ... Set Black Level
        ptrBlackLevel->SetValue(black_level);
        double blackLevelToSet = static_cast<double>(ptrBlackLevel->GetValue());
        // cout << "Black Level to be set to " << blackLevelToSet << "..." << endl;
    }
    catch (Exception& e) {
        cout << "Error Configuring Black Level: " << e.what() << endl;
        result = false;
    }
    // Get Ptr to Node Black Level to Check New Value
    ptrBlackLevel = nodeMap.GetNode("BlackLevel");
    blackLevelToSet = static_cast<double>(ptrBlackLevel->GetValue());
    cout << "  New Black Level: " << blackLevelToSet << endl;

    return result;
}

bool configure(CameraPtr pCam, INodeMap& nodeMap, std::string cameraName, double fps, int width, 
                int height, int offset_x, int offset_y, std::string pixelFormat, int bin_size, 
                double exposure_compensation, double exposure_time, double gain, double black_level, 
                bool hardwareSync, bool primary) {
    bool result = true;
    try {
        // Get the device serial number
        const string deviceSerialNumber = GetDeviceSerial(pCam);
        // Add a new entry into the camera grab info map using the default GrabInfo constructor
        std::cout << "Configuring device " << deviceSerialNumber << endl;
        // set pixel format
        // if (!setPixelFormat(nodeMap, pixelFormat)) {
        //     return false;
        // }
        // Set acquisition mode to continuous
        if (!setContinuousAcquisitionMode(nodeMap)) {
            return false;
        }
        // set fps
        if ((!hardwareSync) || (hardwareSync && primary)) {
            if (!setFps(nodeMap, fps, cameraName)) {
                return false;
            }
        }
        // set resolution
        if (!setResolution(nodeMap, width, height, offset_x, offset_y)) {
            return false;
        }
        // set binning
        if (!setBinning(nodeMap, bin_size)) {
            return false;
        }
        // set exposure compensation
        if(cameraName == "Grasshopper3") {
            if (!setExposureCompensation(nodeMap, exposure_compensation)) {
                return false;
            }
        }
        // set exposure time
        if (!setExposureTime(nodeMap, exposure_time)) {
            return false;
        }
        // set gain
        if (!setGain(nodeMap, gain)) {
            return false;
        }
        // set black level
        if (!setBlackLevel(nodeMap, black_level)) {
            return false;
        }
    }
    catch (Exception& e) {
        std::cout << "Error Configuring camera: " << e.what() << endl;
        result = false;
    }
    return result;
}

// This function acquires and saves 10 images from a device.
int AcquireImages(CameraPtr pCam, INodeMap& nodeMap, INodeMap& nodeMapTLDevice, int num_frames) {
    int result = 0;
    const string deviceSerialNumber = GetDeviceSerial(pCam);
    std::cout << endl << endl << "*** IMAGE ACQUISITION ***" << endl;
    try {
        // Begin acquiring images
        pCam->BeginAcquisition();
        std::cout << "Acquiring images..." << endl;
        // Retrieve, convert, and save images
        const unsigned int k_numImages = num_frames;
        for (unsigned int imageCnt = 0; imageCnt < k_numImages; imageCnt++) {
            try {
                // Retrieve next received image
                ImagePtr pResultImage = pCam->GetNextImage(1000);
                // Ensure image completion
                if (pResultImage->IsIncomplete()) {
                    // Retrieve and print the image status description
                    std::cout << "Image incomplete: " << Image::GetImageStatusDescription(pResultImage->GetImageStatus())
                         << "..." << endl
                         << endl;
                }
                else {
                    // Convert image to mono 8
                    ImagePtr convertedImage = pResultImage->Convert(PixelFormat_Mono8, HQ_LINEAR);
                    // // Retrieve timestamp
                    // ChunkData chunkData = pResultImage->GetChunkData();
                    // uint64_t timestamp = chunkData.GetTimestamp();
                    // // cout << "\tTimestamp: " << timestamp << endl;
                    // Create a unique filename
                    ostringstream filename;
                    filename << "Acquisition-";
                    if (!deviceSerialNumber.empty()) {
                        filename << deviceSerialNumber.c_str() << "-";
                    }
                    filename << imageCnt << ".jpg";
                    // Save image
                    convertedImage->Save(filename.str().c_str());
                    // std::cout << "Image saved at " << filename.str() << endl;
                }
                // Release image
                pResultImage->Release();
                std::cout << endl;
            }
            catch (Spinnaker::Exception& e) {
                std::cout << "Error: " << e.what() << endl;
                result = -1;
            }
        }
        // End acquisition
        pCam->EndAcquisition();
    }
    catch (Spinnaker::Exception& e) {
        std::cout << "Error: " << e.what() << endl;
        return -1;
    }
    return result;
}

CameraPtr getCam(SystemPtr system, CameraList camList, string serialNumber) {
    unsigned int numCameras = camList.GetSize();
    std::cout << "Number of cameras detected: " << numCameras << endl;
    // Finish if there are no cameras
    if (numCameras == 0) {
        camList.Clear();
        system->ReleaseInstance();
        std::cout << "Not enough cameras!" << endl;
    }
    // Configure nir connected camera to use User Set 1 and register image events
    for (unsigned int camIndex = 0; camIndex < camList.GetSize(); camIndex++) {
        CameraPtr cam = camList[camIndex];
        const string deviceSerialNumber = GetDeviceSerial(cam);
        if(string(deviceSerialNumber) == serialNumber) {
            std::cout << "Found!" << endl;
            return cam;
        }
    }
    return nullptr;
}
