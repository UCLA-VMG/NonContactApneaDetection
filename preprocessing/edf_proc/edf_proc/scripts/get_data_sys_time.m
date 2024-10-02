file_path = 'C:\Users\Adnan\Downloads\osa_crop_data\paitent_1_edf\EXN8YAUKXPBRA3TH.edf';
% file_path = 'C:\Users\Adnan\Downloads\osa_crop_data\paitent_1_edf\WIRELESS_SLEEP_STUDY_RESEARCH\EXN8YAPVPCBSDJ4T.edf';
[data_sys_time] = get_nk_signal_2(file_path);

plot(signal_time(1:1000), signal(1:1000))
function [data_sys_time, signal] = get_nk_signal_2(file_path)
    data_sys_time = edfread(file_path, 'DataRecordOutputType','timetable','TimeOutputType','datetime');
    signal_name = "12";
    signal_timetable = vertcat(data_sys_time.(signal_name){:});
    signal = signal_timetable.(signal_name);
    disp(data_sys_time.Properties.VariableNames(signal_name));
end