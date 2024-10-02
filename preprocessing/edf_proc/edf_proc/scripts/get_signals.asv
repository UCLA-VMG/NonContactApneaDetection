%get_signals: function to read signals from an edf
%file_path: path to edf file
%signal_names: array-like containing signal name strings
%returns array containing an array of signals and an array of signal
%timestamps
function [signals, signals_sys_ts] = get_signals(file_path, signal_names)
    data_sys_time = edfread(file_path, 'DataRecordOutputType','timetable','TimeOutputType','datetime'); % read edf

    %initialize cell arrays
    signals{length(signal_names)} = {}; %signals
    signals_sys_ts{length(signal_names)} = {}; %signal timestamps

    for i = 1:length(signal_names) % for every signal name we need
        signal_name = signal_names(i);
       
        signal_timetable = vertcat(data_sys_time.(string(signal_name)){:});
        signal_sys_ts = signal_timetable.Time; %extract signal timestamp
        if signal_name == "AIR_flow"
            signal_name = "AIR-flow";
        end
        signal = signal_timetable.(string(signal_name)); %extract signal

        %append to cell arrays
        signals{i} = signal;
        signals_sys_ts{i} = posixtime(signal_sys_ts); 
    end
end

