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

struct SerialPort : public Sensor
{
    explicit SerialPort(ptree::value_type& sensor_settings, ptree::value_type& global_settings);
    ~SerialPort();

    void AcquireSave(double seconds, boost::barrier& startBarrier);

    int readSerialPort(const char *buffer, unsigned int buf_size);
    bool writeSerialPort(const char *buffer, unsigned int buf_size);
    bool isConnected();
    void closeSerial();
    void signalWriteRead(unsigned int delayTime, std::string command);
    void runBarker13();
    void resetFPGA();

    char incomingData[MAX_DATA_LENGTH];
    std::string portName;
    unsigned int pulseTime, totalTime, baudRate;

    HANDLE _handler;
    bool _connected;
    COMSTAT _status;
    DWORD _errors;
};
