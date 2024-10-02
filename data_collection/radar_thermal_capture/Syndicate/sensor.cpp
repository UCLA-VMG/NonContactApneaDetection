#include "sensor.h"

size_t Sensor::numSensors(0);

Sensor::Sensor(ptree::value_type& sensor_settings, ptree::value_type& global_settings)
    : sensorName(sensor_settings.second.get<std::string>("name")),
    rootPath(global_settings.second.get<std::string>("save_path")),
    bufferSize(0),
    extrinsicMatrix(3, std::vector<double>(4, 0)),
    statusCode(HealthCode::OFFLINE), operatingCode(OpMode::NONE),
    hardwareSync(false), primary(false)
{
    ++numSensors;
    std::filesystem::create_directory(rootPath);
    rootPath = rootPath + sensorName + std::string("/");
    std::filesystem::create_directory(rootPath);
    logFile = std::ofstream(rootPath + "log_" + sensorName + ".txt", std::ios_base::out | std::ios_base::app );

    if(sensor_settings.second.find("hardware_sync") != sensor_settings.second.not_found())
    {
        hardwareSync = sensor_settings.second.get<bool>("hardware_sync");
        std::cout << "Hardware Sync Enabled\n";
    }
    if(sensor_settings.second.find("primary") != sensor_settings.second.not_found())
    {
        primary = sensor_settings.second.get<bool>("primary");
        std::cout << "Primary/Secondary Enabled\n";
    }
}

Sensor::~Sensor()
{
    std::cout << "Sensor " << sensorName << " is shutting down.\n";
}

void Sensor::RecordTimeStamp()
{
    timeStamps.push(std::chrono::system_clock::now());
}

void Sensor::SaveTimeStamps()
{
    std::ofstream timeStampEpochFile(rootPath + "log_timestamps_epoch.txt");
    std::ofstream timeStampFile(rootPath + "log_timestamps.txt");

    while(!timeStamps.empty())
    { 
        // Write to the file
        timeStampEpochFile << timeStamps.front().time_since_epoch().count() << std::endl;  
        // Write in human readable format
        std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(timeStamps.front().time_since_epoch());
        std::chrono::seconds s = std::chrono::duration_cast<std::chrono::seconds>(ms);
        std::time_t t = s.count();
        std::string fractional_seconds = std::to_string(ms.count() % 1000);
        std::tm time_struct = *localtime(&t);
        timeStampFile << std::put_time(&time_struct, "%Y%m%d_%H_%M_%S_")
                      << std::string(3 - std::min(static_cast<size_t>(3), fractional_seconds.length()), '0') 
                      << fractional_seconds << std::endl; 

        timeStamps.pop();      
    }
}


HealthCode Sensor::checkHealthCode()
{
    boost::lock_guard<boost::mutex> guard(mtx_);
    return statusCode;
}

void Sensor::setHealthCode(HealthCode stat)
{
    boost::lock_guard<boost::mutex> guard(mtx_);
    statusCode = stat;
}