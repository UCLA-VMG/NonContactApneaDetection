%only for air-flow

function [signals, signals_sys_ts] = get_signals_airflow(file_path, signal_names, signal_names2)
    data_sys_time = edfread(file_path, 'DataRecordOutputType','timetable','TimeOutputType','datetime'); % read edf

    %initialize cell arrays
    signals{length(signal_names)} = {}; %signals
    signals_sys_ts{length(signal_names)} = {}; %signal timestamps
    signal_names2 = signal_names2(1);
    for i = 1:length(signal_names) % for every signal name we need
        signal_name = signal_names(i);
       
        signal_timetable = vertcat(data_sys_time.(string(signal_name)){:});
        signal_sys_ts = signal_timetable.Time; %extract signal timestamp
        signal = signal_timetable.(string(signal_names2)); %extract signal

        %append to cell arrays
        signals{i} = signal;
        signals_sys_ts{i} = posixtime(signal_sys_ts); 
    end
end