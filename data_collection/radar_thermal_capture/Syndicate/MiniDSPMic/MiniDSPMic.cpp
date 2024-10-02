#include "miniDSPMic.h"

MiniDSPMic::MiniDSPMic(ptree::value_type& sensor_settings, ptree::value_type& global_settings)
    : Sensor(sensor_settings, global_settings), _frames_per_buffer(sensor_settings.second.get<int>("Frames per Buffer"))
{
    _data.fs = sensor_settings.second.get<int>("FS");
    _data.channels = sensor_settings.second.get<int>("Channels");
    PaStreamParameters input_params;
    PaError err = paNoError;

    // Initialize Port Audio
    err = Pa_Initialize();
    if (err != paNoError)
    {
        fprintf(stderr, "An error occurred while using the portaudio stream\n");
        fprintf(stderr, "Error number: %d\n", err);
        fprintf(stderr, "Error message: %s\n", Pa_GetErrorText(err));
        exit(1);
    }
    std::cout << "Initialized Port Audio." << std::endl << std::endl;

    // Get the default input device
    input_params.device = Pa_GetDefaultInputDevice();

    if (input_params.device == paNoDevice)
    {
        fprintf(stderr, "An error occurred while using the portaudio stream\n");
        fprintf(stderr, "Error number: %d\n", err);
        fprintf(stderr, "Error message: %s\n", Pa_GetErrorText(err));
        exit(1);
    }
    const PaDeviceInfo* device_info = Pa_GetDeviceInfo(input_params.device);

    std::cout << "Name                  = " << device_info->name  << std::endl;
    std::cout << "Host API              = " << Pa_GetHostApiInfo(device_info->hostApi)->name  << std::endl;
    std::cout << "Max inputs            = " << device_info->maxInputChannels  << std::endl;
    std::cout << "Max outputs           = " << device_info->maxOutputChannels  << std::endl;
    std::cout << "Default sampling rate = " << device_info->defaultSampleRate       << std::endl;

    input_params.channelCount = _data.channels;                    /* stereo input */
    input_params.sampleFormat = paFloat32;
    input_params.suggestedLatency = Pa_GetDeviceInfo(input_params.device)->defaultLowInputLatency;
    input_params.hostApiSpecificStreamInfo = NULL;

    err = Pa_OpenStream(&record_stream, &input_params, NULL, _data.fs, _frames_per_buffer, paClipOff, _callback, &_data);

    if (err != paNoError)
    {
        fprintf(stderr, "An error occurred while using the portaudio stream\n");
        fprintf(stderr, "Error number: %d\n", err);
        fprintf(stderr, "Error message: %s\n", Pa_GetErrorText(err));
        exit(1);
    }

    std::ostringstream filename;
    filename << rootPath << sensorName << ".raw";
    fid = fopen(filename.str().c_str(), "wb");
    if (fid == NULL)
    {
        std::cout << "Could not open " << filename.str() << "." << std::endl;
    }

}

void MiniDSPMic::AcquireSave(double seconds, boost::barrier& startBarrier) {
    int prev_index  = 0;
    _data.max_frame_index = static_cast<int>(seconds) * _data.fs;
    _data.global_frame_index = 0;
    _data.local_frame_index = 0;
    _data.save_index = 0;
    _data.save_bytes  = MIC_SAVE_FACTOR * _frames_per_buffer; // A little over 1 second with MIC_SAVE_FACTOR = 10 and _frames_per_buffer = 512
    _data.buffer_size = MIC_BUFFER_SIZE * _frames_per_buffer;
    std::cout << _data.save_bytes << " " << _data.buffer_size << std::endl;
    _data.audio_samples_1 = (float*)malloc(_data.buffer_size * _data.channels * sizeof(float));

    PaError err = Pa_StartStream(record_stream);
    if (err != paNoError)
    {
        fprintf(stderr, "An error occurred while using the portaudio stream\n");
        fprintf(stderr, "Error number: %d\n", err);
        fprintf(stderr, "Error message: %s\n", Pa_GetErrorText(err));
        exit(1);
    }
    std::cout << "---------------- Now recording!! Please speak into the microphone. ----------------" << std::endl << std::endl; 

    startBarrier.wait();
    while ((err = Pa_IsStreamActive(record_stream)) == 1)
    {
        Pa_Sleep(MIC_WAIT_MILLS);
        std::cout << "Frame = " << _data.global_frame_index << "\n"; 
        if (prev_index != _data.save_index)
        {
            std::cout << "Start " << _data.save_index << std::endl;
            fwrite(_data.audio_samples_1 + (prev_index * _data.save_bytes * _data.channels), _data.channels * sizeof(float), _data.save_bytes, fid);
            std::cout << "End   " << _data.save_index << std::endl;
            prev_index = _data.save_index;
        }
    }
    // Need to account for the final samples being dropped.
    // fwrite(_data.audio_samples_1 + (prev_index * _data.save_bytes * _data.channels), _data.channels * sizeof(float), _data.save_bytes, fid);
    if (err != paNoError)
    {
        fprintf(stderr, "An error occurred while using the portaudio stream\n");
        fprintf(stderr, "Error number: %d\n", err);
        fprintf(stderr, "Error message: %s\n", Pa_GetErrorText(err));
        exit(1);
    }

    err = Pa_CloseStream(record_stream);

    if (err != paNoError)
    {
        fprintf(stderr, "An error occurred while using the portaudio stream\n");
        fprintf(stderr, "Error number: %d\n", err);
        fprintf(stderr, "Error message: %s\n", Pa_GetErrorText(err));
        exit(1);
    }

    // Save the recording 
    fclose(fid);
    std::cout << "Data has been written." << std::endl;
}

static int _callback(const void* input_buffer, void* output_buffer,
    unsigned long frames_per_buffer, const PaStreamCallbackTimeInfo* time_info,
    PaStreamCallbackFlags status_flags, void* user_data) {
    paRecordData* data = (paRecordData*)user_data;
    const float* read_ptr = (const float*)input_buffer;
    float* write_ptr = &data->audio_samples_1[data->local_frame_index * data->channels];
    long frames_to_calc;
    int finished;
    unsigned long frames_remaining = data->max_frame_index - data->global_frame_index;

    /* Prevent unused variable warnings. */
    (void)output_buffer;
    (void)time_info;
    (void)status_flags;
    (void)user_data;
    if (frames_remaining < frames_per_buffer) {
        frames_to_calc = frames_remaining;
        std::cout << "\nWe are done\n";
        finished = paComplete;
    }
    else {
        frames_to_calc = frames_per_buffer;
        finished = paContinue;
    }

    if (input_buffer == NULL) {
        for (int i = 0; i < frames_to_calc; i++) {
            for (int j = 0; j < data->channels; j++) {
                *write_ptr++ = 0.0f; // Silence
            }
        }
    }
    else {
        for (int i = 0; i < frames_to_calc; i++) {
            for (int j = 0; j < data->channels; j++) {
                *write_ptr++ = *read_ptr++;
            }
        }
    }
    data->global_frame_index += frames_to_calc;
    data->local_frame_index = (data->local_frame_index + frames_to_calc) % data->buffer_size;
    data->save_index = data->local_frame_index / data->save_bytes;  
    return finished;
}

MiniDSPMic::~MiniDSPMic() {
    Pa_Terminate();
    if (_data.audio_samples_1) free(_data.audio_samples_1);
    if (_data.audio_samples_2) free(_data.audio_samples_2);
    // if (err != paNoError)
    // {
    //     fprintf(stderr, "An error occurred while using the portaudio stream\n");
    //     fprintf(stderr, "Error number: %d\n", err);
    //     fprintf(stderr, "Error message: %s\n", Pa_GetErrorText(err));
    // }
}
