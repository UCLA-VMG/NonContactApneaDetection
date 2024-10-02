#pragma once
#include "sensor.h"
#include "portaudio.h"

#include <chrono>
#include <thread>
#include <string>
#include <memory>
#include <any>

#define MIC_WAIT_MILLS 100
#define MIC_SAVE_FACTOR 100
#define MIC_BUFFER_SIZE (10*MIC_SAVE_FACTOR)

typedef struct
{
    int     fs;
    int     channels;
    int     save_index;
    int     save_bytes;
    int     buffer_size;
    int     local_frame_index;  /* Index into sample array. */
    int     global_frame_index;  /* Index into sample array. */
    int     max_frame_index;
    float* audio_samples_1;
    float* audio_samples_2;
}
paRecordData;

struct MiniDSPMic : public Sensor
{
    MiniDSPMic(ptree::value_type& sensor_settings, ptree::value_type& global_settings);
    ~MiniDSPMic();

    void AcquireSave(double seconds, boost::barrier& startBarrier);

    int _frames_per_buffer;
    paRecordData _data;
    PaStream* record_stream;
    FILE* fid;

    
};

static int _callback(const void* input_buffer, void* output_buffer,
    unsigned long frames_per_buffer, const PaStreamCallbackTimeInfo* time_info,
    PaStreamCallbackFlags status_flags, void* userData);
