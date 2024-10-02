from curses import KEY_SLEFT
import numpy as np
import os
from scipy import interpolate
from six import string_types
import sys
import time
import csv
from datetime import datetime
from typing import Callable, List
from collections import MutableMapping
from contextlib import suppress
import warnings
import matplotlib.pyplot as plt

import utils.utils_dict as dict
import utils.utils_read as read
import utils.utils_write as write
import utils.utils_display as display
from data.signal_data import SignalData

# import matlab
# import matlab.engine

import numpy as np
import os
import scipy
from scipy import interpolate
import matplotlib.pyplot as plt
import json
from six import string_types

import sys
import os
import time
import csv
from datetime import datetime

import pickle 

###################################################### Phillips MX800 Help Functions for processing vital sign data

# fps_dict =   [{"NOM_PLETHWaveExport": 125},
#                         {"NOM_RESPWaveExport": 60},
#                         {"NOM_ECG_ELEC_POTL_AVFWaveExport": 750},
#                         {"NOM_ECG_ELEC_POTL_IIWaveExport": 250},
#                         {"NOM_ECG_ELEC_POTL_MCLWaveExport": 500},
#                         {"NOM_ECG_ELEC_POTL_VWaveExport": 750},    
#                         {"NOM_TEMP": 1},
#                         {"NOM_ECG_CARD_BEAT_RATE": 1},
#                         {"NOM_ECG_AMPL_ST_I": 1},
#                         {"NOM_ECG_AMPL_ST_II": 1},
#                         {"NOM_ECG_AMPL_ST_III": 1},
#                         {"NOM_ECG_AMPL_ST_AVR": 1},
#                         {"NOM_ECG_AMPL_ST_AVL": 1},
#                         {"NOM_ECG_AMPL_ST_AVF": 1},
#                         {"NOM_ECG_AMPL_ST_V": 1},
#                         {"NOM_ECG_AMPL_ST_MCL": 1},
#                         {"NOM_PULS_OXIM_SAT_O2": 1},
#                         {"NOM_PLETH_PULS_RATE": 1},
#                         {"NOM_PULS_OXIM_PERF_REL": 1},
#                         {"NOM_PRESS_BLD_NONINV_SYS": 1},
#                         {"NOM_PRESS_BLD_NONINV_DIA": 1},
#                         {"NOM_PRESS_BLD_NONINV_MEA": 1}]


mx800_vital_params =   {"NOM_PLETHWaveExport": {"fps": 125, "chunk": 32},
                        "NOM_RESPWaveExport": {"fps": 60, "chunk": 16},
                        "NOM_ECG_ELEC_POTL_AVFWaveExport": {"fps": 750, "chunk": 192},
                        "NOM_ECG_ELEC_POTL_IIWaveExport": {"fps": 250, "chunk": 64},
                        "NOM_ECG_ELEC_POTL_MCLWaveExport": {"fps": 500, "chunk": 128},
                        "NOM_ECG_ELEC_POTL_VWaveExport": {"fps": 750, "chunk": 192},
                        "NOM_TEMP": {"fps": 1, "chunk": 192},
                        "NOM_ECG_CARD_BEAT_RATE": {"fps": 1, "chunk": 192},
                        "NOM_ECG_AMPL_ST_I": {"fps": 1, "chunk": 192},
                        "NOM_ECG_AMPL_ST_II": {"fps": 1, "chunk": 192},
                        "NOM_ECG_AMPL_ST_III": {"fps": 1, "chunk": 192},
                        "NOM_ECG_AMPL_ST_AVR": {"fps": 1, "chunk": 192},
                        "NOM_ECG_AMPL_ST_AVL": {"fps": 1, "chunk": 192},
                        "NOM_ECG_AMPL_ST_AVF": {"fps": 1, "chunk": 192},
                        "NOM_ECG_AMPL_ST_V": {"fps": 1, "chunk": 192},
                        "NOM_ECG_AMPL_ST_MCL": {"fps": 1, "chunk": 192},
                        "NOM_PULS_OXIM_SAT_O2": {"fps": 1, "chunk": 192},
                        "NOM_PLETH_PULS_RATE": {"fps": 1, "chunk": 192},
                        "NOM_PULS_OXIM_PERF_REL": {"fps": 1, "chunk": 192},
                        "NOM_PRESS_BLD_NONINV_SYS": {"fps": 1, "chunk": 192},
                        "NOM_PRESS_BLD_NONINV_DIA": {"fps": 1, "chunk": 192},
                        "NOM_PRESS_BLD_NONINV_MEA": {"fps": 1, "chunk": 192}}

###################################################### 

def extract_data(input_filepath : str, vital_sign: int):
    file = open(input_filepath, 'r', encoding='utf-8-sig')
    csvreader = csv.reader(file)
    stamp_list = []
    col_idx = 3
    
    path = os.path.dirname(input_filepath)
    filename_ext = os.path.basename(input_filepath)
    filename = os.path.splitext(filename_ext)[0]

    if(filename == "MPDataExport"):
        # This skips the first row of the CSV file.
        next(csvreader)
        col_idx = 2+vital_sign
    
    for i in csvreader:
        stamp_list.append(i)

    sensor_stamps = []
    pc_stamps = []
    data = []

    for j in range(len(stamp_list)):
        time_obj_mx = datetime.strptime(stamp_list[j][0], '%d-%m-%Y %H:%M:%S.%f')
        stamp_mx = time_obj_mx.timestamp()
        sensor_stamps.append(stamp_mx)

        time_obj_sys = datetime.strptime(stamp_list[j][2], '%d-%m-%Y %H:%M:%S.%f')
        stamp_sys = time_obj_sys.timestamp()
        pc_stamps.append(stamp_sys)

        try:
            data.append(float(stamp_list[j][col_idx]))
        except:
            data.append(-1)

    return sensor_stamps, pc_stamps, data
  
###################################################### read vital_dict

### Not processed
# {sensor : {vital_sign : waveform, vital_sign2 : waveform2, ...}, sensor2 : ...}
def read_vital_dict_from_raw(read_dirpath: str, files: list) -> dict:
    vital_dict = {}
    vital_keys = list(mx800_vital_params.keys())
    for i in range(len(vital_keys)):
        vital_key = vital_keys[i]
        unroll_flag = 1
        if (i < 6):
            file = vital_key+".csv"
            file_path = os.path.join(read_dirpath, file)
            if(file in files):
                sensor_stamps, pc_stamps, data = extract_data(file_path, 1)
                vital_dict[vital_key] = [sensor_stamps, pc_stamps, data, unroll_flag, mx800_vital_params[vital_key]["fps"]]    
        else:
            file = "MPDataExport.csv"
            file_path = os.path.join(read_dirpath, file)
            if(file in [files]):
                unroll_flag = 0
                sensor_stamps, pc_stamps, data = extract_data(file_path, i+1)
                vital_dict[vital_key] = [sensor_stamps, pc_stamps, data, unroll_flag, mx800_vital_params[vital_key]["fps"]]    
    return vital_dict

### Assuming it's already inteprolated
def read_vital_dict_from_npy(read_dirpath: str, files: list) -> dict:
    # check only one file
    if(len(files) != 1):
        raise Exception("Expected len of files var = 1, received len = ", len(files))
    file = files[0]
    file_path = os.path.join(read_dirpath, file)
    vital_dict = read.read_npy(file_path)
    return vital_dict

### Assuming it's already inteprolated
def read_vital_dict_from_pkl(read_dirpath: str, files: list) -> dict:
    # check only one file
    if(len(files) != 1):
        raise Exception("Expected len of files var = 1, received len = ", len(files))
    file = files[0]
    file_path = os.path.join(read_dirpath, file)
    vital_dict = read.read_pkl(file_path)
    return vital_dict

def read_philips_vitals(read_dirpath: str, read_files: list, read_vitals: list, offsets: dict, format: str) -> dict:
    if (format == "raw"):
        original_vital_dict = read_vital_dict_from_raw(read_dirpath, read_files)
    else: # if format is processed 
        if(any(".npy" in f for f in read_files)):
            original_vital_dict = read_vital_dict_from_npy(read_dirpath, read_files)
        elif(any(".pkl" in f for f in read_files)):
            original_vital_dict = read_vital_dict_from_pkl(read_dirpath, read_files)
    
    # to account for MMFair dataset format
    if "rgbd" in original_vital_dict.keys(): 
        original_vital_dict = original_vital_dict["rgbd"]
    
    # first check that keys in original and interpolated vitals are identical 
    vitals = {}
    # create dict of signal_data objects for each vital in original_vitals
    for vital in original_vital_dict.keys():
        if (read_vitals) and (vital not in read_vitals):
            continue
        vitals[vital] = SignalData()
        # set sensor_stamps, pc_stamps, data, unroll_flag for signal
        vitals[vital].sensor_stamps, vitals[vital].pc_stamps, vitals[vital].data, vitals[vital].unroll_flag, vitals[vital].fps_data = original_vital_dict[vital]
        vitals[vital].sensor_stamps = np.asarray(vitals[vital].sensor_stamps)
        vitals[vital].pc_stamps = np.asarray(vitals[vital].pc_stamps)
        vitals[vital].data = np.asarray(vitals[vital].data)
        vitals[vital].unroll_flag = np.asarray([vitals[vital].unroll_flag])
        # set offset attribute for signal
        if vital in offsets.keys():
            vitals[vital].offset = offsets[vital]
        else:
            vitals[vital].offset = 25/30
    return vitals

def read_nihon_kohden_vitals(read_dirpath: str, read_files: list) -> dict:
    # nk_vitals = ["CHEST",
    #     "ABD",
    #     "LOC",
    #     "ROC",
    #     "EKG",
    #     "SaO2",
    #     "PTAF",
    #     "O2-M1",
    #     "C2-M1",
    #     "F4-M1",
    #     "R-R",
    #     "CHIN",
    #     "L LEG",
    #     "R LEG",
    #     "ImagingResearch",
    #     "PLETH"]
    # nk_vitals_fps = {"CHEST": 0,
    #     "ABD": 0,
    #     "LOC": 0,
    #     "ROC": 0,
    #     "EKG": 0,
    #     "SaO2": 0,
    #     "PTAF": 0,
    #     "O2-M1": 0,
    #     "C2-M1": 0,
    #     "F4-M1": 0,
    #     "R-R": 0,
    #     "CHIN": 0,
    #     "L LEG": 0,
    #     "R LEG": 0,
    #     "ImagingResearch": 0,
    #     "PLETH": 0}
    # # check only one file
    # if(len(read_files) != 1):
    #     raise Exception("Expected len of files var = 1, received len = ", len(read_files))
    # file = read_files[0]
    # filename, ext = file.split(".")
    # ext = "." + ext
    # file_path = os.path.join(read_dirpath, file)
    # if (ext in [".edf"]):
    #     raise Exception("Expected ext of file = '.edf', receieved ext = ", ext)
    # eng =  matlab.engine.start_matlab()
    # data_dict = eng.proc_nihon_kohden_vitals(file_path)
    # vitals = {}
    # for vital in data_dict.keys():
    #     vitals[vital] = SignalData()
    #     vitals[vital].data = data_dict[vital][:, 0]
    #     vitals[vital].time = data_dict[vital][:, 1]
    #     vitals[vital].fps = nk_vitals_fps[vital]
    # eng.quit()
    # return vitals
    vitals = {}
    return vitals

def read_vitals_data(read_dirpath: str, read_files: list, read_vitals: list, offsets: dict, sensor_type: str, format: str) -> dict:
    vitals = {}
    if (sensor_type in ["Nihon_Kohden", "nihon_kohden", "nk"]):
        vitals = read_nihon_kohden_vitals(read_dirpath, read_files, format)
    elif(sensor_type in ["Philips", "philips", "MX800", "mx800", "Intellivue", "intellivue"]):
        vitals = read_philips_vitals(read_dirpath, read_files, read_vitals, offsets, format)
    return vitals

###################################################### display vital signs individually in vital_dict

def display_vitals_data(vitals: dict, keys: list, display_attributes: list) -> bool:
    if not keys:
        keys = vitals.keys()
    for key in keys:
        if not display_attributes:
            if(vitals[key].data is not None):
                display_attributes.append("data")
            if(vitals[key].interpolated is not None):
                display_attributes.append("interpolated")
            if(vitals[key].time is not None):
                display_attributes.append("time")
            # if(vitals[key].fps is not None):
            #     display_attributes.append("fps")
        for attr in display_attributes:
            if(attr in ["data"]):
                display.display_signal(vitals[key].data, title = key+"_"+attr)
            if(attr in ["interpolated"]):
                display.display_signal(vitals[key].interpolated, title = key+"_"+attr)
            if(attr in ["time"]):
                display.display_signal(vitals[key].time, title = key+"_"+attr)
            # if(attr in ["fps"]):
            #     display.display_signal(vitals[key].fps, title = key+"_"+attr)
    return True

###################################################### write vital_dict and individual or multiple seperate vital signs

def write_vitals_as_imgs(write_dirpath: str, write_files: list, vitals: dict, write_attributes: list) -> bool:
    status_list = []
    # check at least one file
    if(not write_files):
        raise Exception("Expected len of files var > 1, received len = ", len(write_files))
    for file in write_files:
        filename, ext = file.split(".")
        if(filename not in vitals.keys()):
            raise Exception("Expected filename to be one of: ", vitals.keys(), " received filename = ", filename)
        ext = "." + ext
        if(ext not in [".png", ".jpg", ".jpeg"]):
            raise Exception("Expected ext to be one of: ['.png', '.jpg', '.jpeg'], received ext = ", ext)  
        if (not write_attributes):
            if(vitals[filename].data is not None):
                write_attributes.append("data")
            if(vitals[filename].interpolated is not None):
                write_attributes.append("interpolated")
            if(vitals[filename].time is not None):
                write_attributes.append("time")
            # if(vitals[filename].fps is not None):
            #     write_attributes.append("fps")
            if(vitals[filename].sensor_stamps is not None):
                write_attributes.append("sensor_stamps")
            if(vitals[filename].pc_stamps is not None):
                write_attributes.append("pc_stamps")
        for attr in write_attributes:
                if (attr in ["data"]):
                    file_path = os.path.join(write_dirpath, filename + "_" + attr + ext)
                    status = write.write_signal_as_img(file_path, vitals[filename].data, filename)
                    status_list.append(status)
                if (attr in ["interpolated"]):
                    file_path = os.path.join(write_dirpath, filename + "_" + attr + ext)
                    status = write.write_signal_as_img(file_path, vitals[filename].interpolated, filename)
                    status_list.append(status)
                # if (attr in ["time"]):
                #     status = write.write_signal_as_img(file_path, vitals[filename].time, filename)
                # if (attr in ["fps"]):
                #     status = write.write_signal_as_img(file_path, vitals[filename].fps, filename)        
    status = all(status_list)
    return status
    
def write_vitals_as_arrays(write_dirpath: str, write_files: list, vitals: dict, write_attributes: list) -> bool:
    status_list = []
    # check at least one file
    if(not write_files):
        raise Exception("Expected len of files var >= 1, received len = ", len(write_files))
    for file in write_files:
        filename, ext = file.split(".")
        if(filename not in vitals.keys()):
            raise Exception("Expected filename to be one of: ", vitals.keys(), " received filename = ", filename)
        ext = "." + ext
        if(ext not in [".npy", ".dat"]):
            raise Exception("Expected ext to be one of: ['.png', '.jpg', '.jpeg'], received ext = ", ext)
        if (not write_attributes):
            if(vitals[filename].data is not None):
                write_attributes.append("data")
            if(vitals[filename].interpolated is not None):
                write_attributes.append("interpolated")
            if(vitals[filename].time is not None):
                write_attributes.append("time")
            # if(vitals[filename].fps is not None):
            #     write_attributes.append("fps")
            if(vitals[filename].sensor_stamps is not None):
                write_attributes.append("sensor_stamps")
            if(vitals[filename].pc_stamps is not None):
                write_attributes.append("pc_stamps")
        if(ext in [".dat"]):
            for attr in write_attributes:
                if (attr in ["data"]):
                    file_path = os.path.join(write_dirpath, filename + "_" + attr + ext)
                    status = write.write_dat(file_path, vitals[filename].data)
                    status_list.append(status)
                if (attr in ["interpolated"]):
                    file_path = os.path.join(write_dirpath, filename + "_" + attr + ext)
                    status = write.write_dat(file_path, vitals[filename].interpolated)
                    status_list.append(status)
                if (attr in ["time"]):
                    file_path = os.path.join(write_dirpath, filename + "_" + attr + ext)
                    status = write.write_dat(file_path, vitals[filename].time)
                    status_list.append(status)
                # if (attr in ["fps"]):
                #     file_path = os.path.join(write_dirpath, filename + "_" + attr + ext)
                #     status = write.write_dat(file_path, vitals[filename].fps)
                #     status_list.append(status)
                if (attr in ["sensor_stamps"]):
                    file_path = os.path.join(write_dirpath, filename + "_" + attr + ext)
                    status = write.write_dat(file_path, vitals[filename].sensor_stamps)
                    status_list.append(status)
                if (attr in ["pc_stamps"]):
                    file_path = os.path.join(write_dirpath, filename + "_" + attr + ext)
                    status = write.write_dat(file_path, vitals[filename].pc_stamps)
                    status_list.append(status)
        elif(ext in [".npy"]):
            for attr in write_attributes:
                if (attr in ["data"]):
                    file_path = os.path.join(write_dirpath, filename + "_" + attr + ext)
                    status = write.write_npy(file_path, vitals[filename].data)
                    status_list.append(status)
                if (attr in ["interpolated"]):
                    file_path = os.path.join(write_dirpath, filename + "_" + attr + ext)
                    status = write.write_npy(file_path, vitals[filename].interpolated)
                    status_list.append(status)
                if (attr in ["time"]):
                    file_path = os.path.join(write_dirpath, filename + "_" + attr + ext)
                    status = write.write_npy(file_path, vitals[filename].time)
                    status_list.append(status)
                # if (attr in ["fps"]):
                #     file_path = os.path.join(write_dirpath, filename + "_" + attr + ext)
                #     status = write.write_npy(file_path, vitals[filename].fps)
                #     status_list.append(status)
                if (attr in ["sensor_stamps"]):
                    file_path = os.path.join(write_dirpath, filename + "_" + attr + ext)
                    status = write.write_npy(file_path, vitals[filename].sensor_stamps)
                    status_list.append(status)
                if (attr in ["pc_stamps"]):
                    file_path = os.path.join(write_dirpath, filename + "_" + attr + ext)
                    status = write.write_npy(file_path, vitals[filename].pc_stamps)
                    status_list.append(status)
    status = all(status_list)
    return status

def write_vitals_as_dict(write_dirpath: str, write_files: list, vitals: dict, write_attributes: list) -> bool:
    if(len(write_files) != 1):
        raise Exception("Expected len of files var = 1, received len = ", len(write_files))
    file = write_files[0]
    file_path = os.path.join(write_dirpath, file)
    vital_dict = {}
    for key in vitals.keys():
        vital_dict[key] = {}
        if (not write_attributes):
            if(vitals[key].data is not None):
                write_attributes.append("data")
            if(vitals[key].interpolated is not None):
                write_attributes.append("interpolated")
            if(vitals[key].time is not None):
                write_attributes.append("time")
            # if(vitals[key].fps is not None):
            #     write_attributes.append("fps")
            if(vitals[key].sensor_stamps is not None):
                write_attributes.append("sensor_stamps")
            if(vitals[key].pc_stamps is not None):
                write_attributes.append("pc_stamps")
        if (len(write_attributes) == 1):
            if (write_attributes[0] in ["data"]):
                vital_dict[key]= vitals[key].data
            elif (write_attributes[0] in ["interpolated"]):
                vital_dict[key]= vitals[key].interpolated
            elif (write_attributes[0] in ["time"]):
                vital_dict[key]= vitals[key].time
            # elif (write_attributes[0] in ["fps"]):
            #     vital_dict[key]= vitals[key].fps
            elif (write_attributes[0] in ["sensor_stamps"]):
                vital_dict[key]= vitals[key].sensor_stamps
            elif (write_attributes[0] in ["pc_stamps"]):
                vital_dict[key]= vitals[key].pc_stamps
        else:
            for attr in write_attributes:
                if (attr in ["data"]):
                    vital_dict[key]["data"] = vitals[key].data
                if (attr in ["interpolated"]):
                    vital_dict[key]["interpolated"] = vitals[key].interpolated
                if (attr in ["time"]):
                    vital_dict[key]["time"] = vitals[key].time
                # if (attr in ["fps"]):
                #     vital_dict[key]["fps"] = vitals[key].fps
                if (attr in ["sensor_stamps"]):
                    vital_dict[key]["sensor_stamps"] = vitals[key].sensor_stamps
                if (attr in ["pc_stamps"]):
                    vital_dict[key]["pc_stamps"] = vitals[key].pc_stamps
    status = write.write_pkl(file_path, vital_dict)
    return status

def write_vitals_data(write_dirpath: str, write_files: list, vitals: dict, write_attributes: list, format: str) -> bool:
    if(format in ["img", "imgs"]):
        status = write_vitals_as_imgs(write_dirpath, write_files, vitals, write_attributes)
    if(format in ["array", "arrays"]):
        status = write_vitals_as_arrays(write_dirpath, write_files, vitals, write_attributes)
    if(format in ["dict"]):
        status = write_vitals_as_dict(write_dirpath, write_files, vitals, write_attributes)
    return status
    
###################################################### align vitals

# # vital_file = "vitals.edf"
# # sent_sync_signal_file = "LOG_FILE.txt"

# # def get_sent_sync_signal(file_path) -> np.ndarray:
# #     sent_sync_signal = np.empty((1,1))
# #     return sent_sync_signal

# # def get_received_sync_signal(file_path) -> np.ndarray:
# #     received_sync_signal = np.empty((1,1))
# #     return received_sync_signal

# # def get_sync_offset(sent_sync_signal, received_sync_signal) -> int:
# #     sync_offset = 0
# #     return sync_offset

# def align_vitals_data(vitals):
#     vital_signs = ["CHEST",
#         "ABD",
#         "LOC",
#         "ROC",
#         "EKG",
#         "SpO2",
#         "PTAF",
#         "O2-M1",
#         "C2-M1",
#         "F4-M1",
#         "R-R",
#         "CHIN",
#         "L LEG",
#         "R LEG",
#         "IMAGING"]
#     # calculate offset

#     # align signals
#     offset = sync_offset * fps
#     return vitals

################################################################################ transforming data

############################################ interpolating data

def extract_timestamps(filename):
    file = open(filename, 'r')
    data = file.readlines()

    ts = []
    for i in data:
        time_obj = datetime.strptime(i.rstrip(), '%Y%m%d_%H_%M_%S_%f')
        ts.append(time_obj.timestamp())
    
    return np.array(ts)
      
def find_deltas(pc_stamps, sensor_stamps):
    const_num = pc_stamps[0]
    deltas = []
    for i, stamp in enumerate(pc_stamps):
        if(stamp != const_num):
            const_num = stamp
            delta = pc_stamps[i-1] - sensor_stamps[i-1]
            if(pc_stamps[i] - pc_stamps[i-1] > 0.3): # comparison for blips
                deltas.append(delta)
    return np.array(deltas)

def find_deltas2(pc_stamps, sensor_stamps):
    pc_stamps = pc_stamps[0:200*32]
    sensor_stamps  = sensor_stamps[0:200*32]

    const_num = pc_stamps[0]
    deltas = []
    for i, stamp in enumerate(pc_stamps):
        if(stamp != const_num):
            const_num = stamp
            delta = pc_stamps[i-1] - sensor_stamps[i-1]
            if(pc_stamps[i] - pc_stamps[i-1] > 0.3): # comparison for blips
                deltas.append(delta)
    return np.array(deltas)

def unroll_stamps(sensor_stamps, batch_size = int(32), time_diff = 0.256):
    unrolled_stamps = []
    for i in range(int(len(sensor_stamps)/batch_size)):
        current_stamp = sensor_stamps[i * batch_size]
        for j in range(batch_size):
            unrolled_val = current_stamp - time_diff + time_diff*(j+1)/batch_size
            unrolled_stamps.append(unrolled_val)
    return np.array(unrolled_stamps)

def unroll_stamps2(sensor_stamps, batch_size = int(32*6), time_diff = 0.256):
    unrolled_stamps = []
    current_stamp = sensor_stamps[0] - time_diff

    for i in range(int(len(sensor_stamps)/batch_size)):
        current_stamp += time_diff
        for j in range(batch_size):
            unrolled_val = current_stamp - time_diff + time_diff*(j+1)/batch_size
            unrolled_stamps.append(unrolled_val)
            # if ((i == 0) and (j == 0)) or ((i == int(len(sensor_stamps)/batch_size) - 1) and (j == batch_size - 1)):
            #     print(datetime.fromtimestamp(unrolled_val).strftime("%d-%m-%Y %H:%M:%S.%f"))
    return np.array(unrolled_stamps)

def apply_delta(sensor_stamps, sys_mx_time_delta):
    return sensor_stamps + sys_mx_time_delta

def timestamp_process(ts):
        f = ((float(ts)/1e6)-int(float(ts)/1e6))*1e6

        ts = int(float(ts)/1e6)
        s = ((float(ts)/1e2)-int(float(ts)/1e2))*1e2
        ts = int(float(ts)/1e2)
        m = ((float(ts)/1e2)-int(float(ts)/1e2))*1e2
        ts = int(float(ts)/1e2)
        h = ((float(ts)/1e2)-int(float(ts)/1e2))*1e2

        temp = (3600*h)+(60*m)+s+(f*1e-6)
        temp = float(int(temp*1e6))/1e6

        return temp

def interpolate_signal(data, data_ts, ref_ts):     
    ##CHECK FOR Data AND TS LENGTHS AND CORRECT
    l1 = len(data_ts)
    l2 = len(data)
    if(l2 - l1 != 0):
        for i in range (l2 - l1):
            data_ts = np.append(data_ts, [0])
    l1 = len(data_ts)
    l2 = len(data)

    if l1<l2:
        raise Exception("Unequal MX800 Data and Timestamp Lengths!")
    elif l2<l1:
        raise Exception("Unequal MX800 Data and Timestamp Lengths!")
    #interpolation function
    f = interpolate.interp1d(data_ts, data, kind='linear')

    reinterp_data = []
    for t_temp in ref_ts:
        if t_temp<data_ts[0]: #If timestamp is before start of MX800 fill with starting values
            reinterp_data.append(data[0])
        elif t_temp>data_ts[-1]: #If timestamp is after end of MX800 fill with ending values
            reinterp_data.append(data[-1])
        else:
            reinterp_data.append(f(t_temp))
    output = np.array(reinterp_data)
    return output

def interpolate_vital_dict(vitals, timestamp_path: str):
    # get reference timestamps
    ref_ts = extract_timestamps(timestamp_path)
    # constucting arrays for the data
    for key in vitals.keys():
        sensor_stamps = vitals[key].sensor_stamps
        pc_stamps = vitals[key].pc_stamps
        data = vitals[key].data
        unroll_flag = vitals[key].unroll_flag
        offset = vitals[key].offset
        delta_array = find_deltas(pc_stamps, sensor_stamps)
        sys_mx_time_delta = np.mean(delta_array)
        if(unroll_flag):
            mx_unrolled = unroll_stamps2(sensor_stamps, mx800_vital_params[key]["chunk"])
        else:
            mx_unrolled = sensor_stamps
        data_ts = apply_delta(mx_unrolled, sys_mx_time_delta)
        data_ts = data_ts - offset
        vitals[key].interpolated = interpolate_signal(data, data_ts, ref_ts)
    return vitals

def aslist_cronly(value):
    if isinstance(value, string_types):
        value = filter(None, [x.strip() for x in value.splitlines()])
    return list(value)

def aslist(value, flatten=True):
    """ Return a list of strings, separating the input based on newlines
    and, if flatten=True (the default), also split on spaces within
    each line."""
    values = aslist_cronly(value)
    if not flatten:
        return values
    result = []
    for value in values:
        subvalues = value.split()
        result.extend(subvalues)
    return result

def interpolate_vitals_data(vitals: dict, read_dirpath: str, ref_ts_file: list, sensor_type: str):
    if (sensor_type in ["Nihon_Kohden", "nihon_kohden", "nk"]):
        vitals = None #TODO
    elif(sensor_type in ["Philips", "philips", "MX800", "mx800", "Intellivue", "intellivue"]):
        if (not ref_ts_file) or (len(ref_ts_file) > 1):
            raise Exception("Expected number of refernce timestamp files = 1, received: ", len(ref_ts_file))
        else:
            ref_ts_file = ref_ts_file[0]
            ref_ts_filepath = os.path.join(read_dirpath, ref_ts_file)
        vitals = interpolate_vital_dict(vitals, ref_ts_filepath)
    return vitals
