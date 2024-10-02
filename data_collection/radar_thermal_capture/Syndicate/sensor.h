#pragma once

#include <iostream>
#include <fstream>
#include <queue>
#include <vector>
#include <memory>
#include <chrono>
#include <ctime>
#include <any>
#include <filesystem>

#include <boost/thread.hpp>
#include <boost/thread/barrier.hpp>
#include <boost/bind.hpp>
#include <boost/atomic.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/foreach.hpp>

using boost::property_tree::ptree;


enum HealthCode
{
    OFFLINE,
    RUNNING,
    BUSY,
    ONLINE
};

enum OpMode
{
    NONE,
    COLLECT,
    INFERENCE
};

struct Sensor
{
    static size_t numSensors;
    std::shared_ptr<Sensor> primarySensor;

    //-------- Sensor Identifier and Operating Status/Mode/Characteristics ------------------------
    //Identifiers
    std::string sensorName;
    size_t sensorIdx;
    //Operating Status/Mode
    HealthCode statusCode;
    OpMode operatingCode;
    //Parallel Processing Constructs
    boost::mutex mtx_; 
    bool hardwareSync;
    bool primary;
    // extrinsicMatrix
    // Usage :: Should be used for converting between frame of references between sensors
    std::vector<std::vector<double>> extrinsicMatrix;
    // timeStamps
    // Usage :: holds timestamp objects correlated to a discrete sensor acquistion 
    // Design Note :: system_clock is used instead of steady_clock with the assumption
    //                that recordings do not last longer than several minutes.
    std::queue<std::chrono::time_point<std::chrono::system_clock>> timeStamps;
    // runningBuffer
    // Usage :: holds discrete packets of data acquired by discrete sensor.
    // Design Note :: <std::any> allows for derived classes to fill the buffer with any
    //                type of data. Additionally, the design is intentionally made to be
    //                a queue for concurrent save (to minimize memory requirements) or
    //                for streaming inference mode.
    std::queue<std::any> runningBuffer; 
    // rootPath
    // Usage :: path to save data from the sensor
    // Design Note :: rootPath is reset to <rootPath + sensorName> in constructor.
    std::string rootPath;
    // bufferSize
    // Usage :: for streaming mode, a bufferSize should be specified to cut off extra data
    const int bufferSize;
    // logFile
    // Usage :: file object used for logging details during acquisition.
    std::ofstream logFile;

    //-------- General Functions -------------------------------------------------------
    Sensor(ptree::value_type& sensor_settings, ptree::value_type& global_settings);

    Sensor(int buffer_size);

    virtual ~Sensor();

    // virtual void LogData();

    void RecordTimeStamp();

    void SaveTimeStamps();

    //-------- Capture Data Functions : operatingCode == OpMode::COLLECT -------------


    virtual void AcquireSave(double seconds, boost::barrier& startBarrier) = 0;
    //-------- Inference Functions : operatingCode == OpMode::INFERENCE ---------
    void ContiniousAcquire(size_t queue_size);

    //-------- Thread Safe Access Function --------------------------------------
    HealthCode checkHealthCode();

    void setHealthCode(HealthCode stat);

};

template<class T>
std::unique_ptr<Sensor> makeSensor(ptree::value_type& sensor_settings, ptree::value_type& global_settings)
{
    std::cout << "Create Sensor\n";
    return std::make_unique<T>(sensor_settings, global_settings);
}