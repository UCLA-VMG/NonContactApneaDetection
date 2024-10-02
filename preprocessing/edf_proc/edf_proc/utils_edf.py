import numpy as np
import scipy
import scipy.signal as sig
from tqdm import tqdm
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
def read_timestamps(timestamps_paths, gmt_diff = 7):
    #open, read, remove newline chars, and convert to float for each path
    stamps = []
    for path in timestamps_paths:
        ts_file = open(path, 'r')
        stamps = stamps + [float(line.strip()) for line in ts_file.readlines()]
    

    #convert to numpy array
    #the input file has resolution of 10^-7 s, we convert to s
    stamps = np.asarray(stamps) / 10000000

    #our timestamps are in GMT, so we convert to local time
    stamps = stamps - gmt_diff * 3600 #gmt is 7 hours ahead of pst
    return stamps

def process_signals(signals_raw, signal_names, sample_rates, trial_length, sync_signal_name, fs, rising_edge_ts = None):

    if rising_edge_ts is None:
        try:
            sync_signal = signals_raw[sync_signal_name][0]
            sync_signal_ts = signals_raw[sync_signal_name][1]
            sync_signal = sync_signal - np.min(sync_signal)
            sync_signal = sync_signal / np.max(sync_signal)
            sync_signal = ((sync_signal[:-1] < 0.5) & (sync_signal[1:] > 0.5)).flatten()
        except:
            print("Synchronization signal missing!")
        rising_edges = np.nonzero(sync_signal == 1)
        rising_edge_ts = sync_signal_ts[rising_edges]


    signals_resampled = dict()
    for key in signals_raw.keys():
        print("Processing signal: ", key)
        signals_resampled[key] = process_single_signal(signals_raw[key], rising_edge_ts,sample_rates[key], fs, trial_length)

    dicts = [{},{},{}]
    for key in signals_resampled.keys():
        for trial in range(3):
            dicts[trial][key] = signals_resampled[key][trial]

    return dicts

def process_single_signal(signal_raw, rising_edge_ts,sample_rate, fs, trial_length):
    num_edges_trial = int(trial_length / 6)
    signals_resampled = np.zeros((3, trial_length))
    signal_full = np.zeros(int(trial_length/30*210))

    trial = 0 # what trial is the current index in?
    idxs = np.searchsorted(signal_raw[1].flatten(), rising_edge_ts).flatten()
    last_idx = 0 # in the signal_full array, what is the last index we accessed?
    trial_begin_idx = 0 # what is the index of the beginning of the current trial?
    for i, idx in tqdm(enumerate(idxs[:-1])):
        end_idx = idxs[i+1]

        signal = signal_raw[0][idx:end_idx].flatten()
        if i == 0:
            signal_full[:end_idx - idx] = signal
            last_idx = end_idx
            trial_begin_idx = idx
        elif rising_edge_ts[i+1] - rising_edge_ts[i] > 60:
            signals_resampled[trial] = sig.resample(signal_full[:last_idx-trial_begin_idx], trial_length)
            trial += 1
            trial_begin_idx = end_idx
        else:
            try:
                signal_full[idx-trial_begin_idx:end_idx-trial_begin_idx] = signal
                last_idx = end_idx
            except:
                print(rising_edge_ts[i] - rising_edge_ts[i-1])
                return
    
    signal_full[end_idx-trial_begin_idx:end_idx-trial_begin_idx+int(sample_rate/5)] = signal_raw[0][end_idx:end_idx+int(sample_rate/5)].flatten()
    resample_length = (len(rising_edge_ts) - 2 * num_edges_trial)*6
    signals_resampled[2][:resample_length] = sig.resample(signal_full[:end_idx-trial_begin_idx+int(sample_rate/5)], resample_length)

    return signals_resampled

# find_nearest: finds the index of the item closest to value in array
def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx



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