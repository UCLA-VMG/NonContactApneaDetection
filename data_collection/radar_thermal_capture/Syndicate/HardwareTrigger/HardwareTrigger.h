#pragma once
#include "sensor.h"

#include <chrono>
#include <thread>
#include <string>
#include <memory>
#include <any>
#include <vector>
#include <iostream>
#include <windows.h>
#include <filesystem>

#define ARDUINO_WAIT_TIME 2000
#define MAX_DATA_LENGTH 255

struct HardwareTrigger : public Sensor
{
    explicit HardwareTrigger(ptree::value_type& sensor_settings, ptree::value_type& global_settings);
    ~HardwareTrigger();

    void AcquireSave(double seconds, boost::barrier& startBarrier);

    int readHardwareTrigger(const char *buffer, unsigned int buf_size);
    bool writeHardwareTrigger(const char *buffer, unsigned int buf_size);
    bool isConnected();
    void closeSerial();
    void signalWriteRead(unsigned int delayTime, std::string command);

    char incomingData[MAX_DATA_LENGTH];
    std::string portName;
    unsigned int frequency, totalTime;

    HANDLE _handler;
    bool _connected;
    COMSTAT _status;
    DWORD _errors;
};
