import numpy as np
import scipy
import scipy.signal as sig
import string
import matlab.engine as mt
import math
from matplotlib import pyplot as plt
import os
import pickle

# matlab_init: initializes matlab engine object and changes directory 
# of MATLAB object to the folder where our script is located
# script_path: path to folder where the MATLAB script to read the edf is located
# returns MATLAB engine object
def matlab_init(script_path):
    eng = mt.start_matlab()
    eng.cd(script_path, nargout = 0)
    return eng

# get_signals: wraps MATLAB script to read edf
# edf_path: path to edf file
# signal_names: array-like containing signal name strings
# eng: MATLAB engine object initialized with script path
# returns dict of numpy arrays containing signals and signal timestamps
# key is signal name, value is tuple of numpy arrays of signal and signal timestamps
def get_signals(edf_path, signal_names, eng):
    #call MATLAB script to read signals
    signals, signals_sys_ts = eng.get_signals(edf_path, signal_names, nargout = 2)
    # signals: list of MATLAB arrays containing signals
    # signals_sys_ts: list of MATLAB arrays containing signal timestamps in posixtime (seconds since UNIX epoch)
    
    #create dictionary for signals
    signals_dict = dict()
    for i, name in enumerate(signal_names): #for all signal names
        #add key of signal name, value of tuple of numpy arrays of signal and signal timestamps
        signals_dict.update({name: (np.asarray(signals[i]), np.asarray(signals_sys_ts[i]))})

    return signals_dict

# read_timestamps: reads timestamps of Barker code start time
# timestamps_paths: list of paths to txt files containing Barker code start times
#                   these files are "Arduino_Serial/log_timestamps_epoch.txt" 
#                   in each trial folder
# returns numpy array of timestamps in posixtime
def read_timestamps(timestamps_paths):
    #open, read, remove newline chars, and convert to float for each path
    stamps = []
    for path in timestamps_paths:
        ts_file = open(path, 'r')
        stamps = stamps + [float(line.strip()) for line in ts_file.readlines()]
    

    #convert to numpy array
    #the input file has resolution of 10^-7 s, we convert to s
    stamps = np.asarray(stamps) / 10000000

    #our timestamps are in GMT, so we convert to local time
    stamps = stamps - 7 * 3600 #gmt is 7 hours ahead of pst
    return stamps

# barker_code: generates barker code signal based on timestamps
# timestamps_paths: list of paths to txt files containing Barker code start times
#                   these files are "Arduino_Serial/log_timestamps_epoch.txt" 
#                   in each trial folder
# stamps_edf: numpy array of timestamps from the edf file, can be from any signal
#             used to make our barker signal occupy the same time window as theirs
#             so that the correlation works correctly
# barker_signal_name: string containing the name of our barker code signal from the edf
# sample_rates: dict of sample rates for signals from the edf
#               key is signal name (string), value is sample rate (int)
# returns numpy array containing barker code signal based on timestamps
def barker_code(timestamps_paths, stamps_edf, barker_signal_name, sample_rates, barkerpulse_sample_length):
    barker_sequence = (1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1) #sequence for the barker code
    sample_rate = sample_rates[barker_signal_name]

    stamps_ours = read_timestamps(timestamps_paths) # read timestamps
    stamp_indices = [int(sample_rate *(s - stamps_edf[0])) for s in stamps_ours]
    signal = np.zeros([int(sample_rate*(stamps_edf[-1] - stamps_edf[0])) + 1, 1]) #make the signal the same size as the edf signal
    
    for i in stamp_indices: #barker code beginning at every timestamp
        for n in range(barkerpulse_sample_length * len(barker_sequence)):
            if (i+n) < signal.shape[0]:
                signal[i+n][0] = barker_sequence[int(n / barkerpulse_sample_length)]
    
    return signal

# process_barker: turns signal from edf into a signal with values in [0,1]
# barker_signal: barker code signal from edf
# returns processed signal
# TODO: address issue with noise - we want 0 and 1 to represent the average values
# of when the signal should be 0 or 1
def process_barker(barker_signal):
    signal = abs(barker_signal) #barker signal is flipped in some edfs
    signal = signal - signal.min() #min of signal should be 0
    signal = signal / signal.max() #max of signal should be 1

    return signal

# find_nearest: finds the index of the item closest to value in array
def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx

# get_sync_delta: computes difference between time in edf and time in our timestamps
# timestamps_paths: list of paths to txt files containing Barker code start times
#                   these files are "Arduino_Serial/log_timestamps_epoch.txt" 
#                   in each trial folder
# stamps_edf: numpy array of timestamps from the edf file, can be from any signal
#             used to make our barker signal occupy the same time window as theirs
#             so that the correlation works correctly
# barker_edf: normalized barker code signal from the edf file
# barker_signal_name: string containing the name of our barker code signal from the edf
# sample_rates: dict of sample rates for signals from the edf
#               key is signal name (string), value is sample rate (int)
def get_sync_delta(timestamps_paths, stamps_edf, barker_edf, barker_signal_name, sample_rates, barkerpulse_sample_length):
    barker_ours = barker_code(timestamps_paths, stamps_edf, barker_signal_name, sample_rates, barkerpulse_sample_length)
    correlation = sig.correlate(barker_edf, barker_ours)
    return (np.argmax(correlation) - len(correlation) / 2) / sample_rates[barker_signal_name]


# process_signals: processes signals from edf to make them ready for machine learning
#                  aligns them with time when our hardware starts
#                  resamples to 30 Hz
# edf_path: path to edf file
# timestamps_paths: list of paths to txt files containing Barker code start times
#                   these files are "Arduino_Serial/log_timestamps_epoch.txt" 
#                   in each trial folder
# script_path: path to folder where the MATLAB script to read the edf is located
# signal_names: array-like containing signal name strings
# sample_rates: dict of sample rates for signals from the edf
#               key is signal name (string), value is sample rate (int)
# barker_signal_name: string, name of barker code signal in edf
# barkerpulse_sample_length: length in samples of the smallest pulse in the barker code signal
# trial_length: number of samples in each trial
# -------------------IMPORTANT--------------------------------------
# time_delay: optional parameter to specify a manually estimated time delay for instances
#             where our recording starts before the edf recording
#             most patient data has this issue so this parameter is important
#             estimate by plotting barker_code as well signals_raw[barker_signal_name]
#             and determining how long in seconds the edf recording is delayed
# -------------------IMPORTANT--------------------------------------
# returns list of dicts of numpy arrays containing signals
# key is signal name, value numpy array containing signal
# each list entry corresponds to a trial in timestamps_paths
def process_signals(signals_raw, timestamps_paths, signal_names, sample_rates, barker_signal_name, barkerpulse_sample_length, trial_length, time_delay = None):

    if barker_signal_name != None:
        barker_edf = process_barker(signals_raw[barker_signal_name][0]) # get barker code signal from edf
    
        

    # difference in seconds between time in edf and time in our timestamps
    if time_delay == None:
        time_delay = get_sync_delta(timestamps_paths, signals_raw[barker_signal_name][1], barker_edf, barker_signal_name, sample_rates, barkerpulse_sample_length) #no zero padding so we have the right value

    print("Time Delay: ", time_delay, "s")

    #for each trial, create a dict of signals for that trial
    signals_dicts = []
    for path in timestamps_paths:
        begin = read_timestamps([path])[0]
        record_time_diff = (begin - signals_raw[signal_names[0]][1][0] + time_delay)[0]
        print("Trial " , path, " started ", record_time_diff, "seconds after the ground truth recording.")
        signals_dict = dict()
        for name in signal_names:
            begin_idx = int(max(0, record_time_diff) * sample_rates[name])
            signals_dict[name] = (signals_raw[name][0])[begin_idx:] #only use data from after the start of the trial
            signals_dict[name] = sig.resample(signals_dict[name], int(len(signals_dict[name]) * 30.0 / sample_rates[name])) #resample to 30 Hz
            signals_dict[name] = signals_dict[name][:trial_length] #don't use data from after the trial ends
        
        signals_dicts = signals_dicts + [signals_dict]

    return signals_dicts

# merge_dicts: merges signal dicts from multiple trials
# signals_dicts: list of signal dicts
# signal_names: array-like containing signal name strings
def merge_dicts(signals_dicts, signal_names):
    signals_dict = dict()
    for name in signal_names:
        for i in range(len(signals_dicts)):
            d = signals_dicts[i]
            if i == 0:
                signal = d[name]
            else:
                signal = np.concatenate((signal, d[name]),axis=0)
        signals_dict[name] = signal

    return signals_dict

# begin_time: in the case that our recording starts before the ground truth 
#             we need to account for this when splitting the dictionaries
# camera_timestamps_path: path to Thermal_Camera/log_timestamps_epoch.txt for
#                         trial ground truth starts during
# time_delay: difference in seconds between time in edf and time in our timestamps
def begin_time(camera_timestamps_path, signals_raw, signal_names, time_delay):
    camera_stamps = read_timestamps([camera_timestamps_path])
    begin_idx = find_nearest(camera_stamps + time_delay, signals_raw[signal_names[0]][1][0][0])
    begin_min = math.ceil(begin_idx / 1800)
    begin_dict_idx = begin_min * 1800 - begin_idx
    return {"min": begin_min, "idx": begin_dict_idx}

# split_dicts: splits dictionary into several dictionaries for training purposes
# signals_dict: original dict of numpy arrays
# signal_names: list of signal names
# num_samples: how long should the resulting dictionaries be
# begin_dict_idx: index of first sample to be saved in folders
# returns a list of dicts of numpy arrays containing ground truth signals num_samples long

def split_dicts(signals_dict, signal_names, num_samples, begin_dict_idx = 0):
    signals_dicts = []
    for i in range(int((len(signals_dict[signal_names[0]]) - begin_dict_idx)/ num_samples)):
        d = dict()
        for name in signal_names:
            d[name] = signals_dict[name][int(i*num_samples + begin_dict_idx):int((i+1)*num_samples + begin_dict_idx)]
            d[name] = np.reshape(d[name], [num_samples,])
        signals_dicts.append(d)
    
    return signals_dicts


# save_dicts: saves pickled ground truth vital dicts in subfolders of save_pkl_path
# all_dicts: list of dicts of numpy arrays containing the ground truth data
# save_pkl_path: path containing our split data
# begin_min: index of first folder to save in
def save_dicts(all_dicts, save_pkl_path, begin_min, patient_number):

    for i in range(len(all_dicts)):
        path = os.path.join(save_pkl_path, "v_" + patient_number + "_" + str(int(i+begin_min)))
        path = os.path.join(path, "ground_truth_dicts.pkl")
        with open(path, 'wb') as handle:
            pickle.dump(all_dicts[i], handle, protocol=pickle.HIGHEST_PROTOCOL)