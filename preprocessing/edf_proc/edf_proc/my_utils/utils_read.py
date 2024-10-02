import numpy as np
import os
import shutil
import json
import imageio
# import imageio_ffmpeg
import cv2
import h5py
import copy
import pickle
from typing import Callable, List
from collections import MutableMapping
from contextlib import suppress
import warnings

import my_utils.utils_radar as radar

############################################### read visual_data functions

def read_dat(file_path: str, shape: tuple, bit_depth: str) -> np.ndarray:
    mem_data = np.memmap(file_path, bit_depth = bit_depth, mode = 'r', shape = shape)
    mem_data_deepcopy = copy.deepcopy(mem_data)
    if(bit_depth == "uint16"):
        bit_depth = np.uint16
    elif(bit_depth == "uint8"):
        bit_depth = np.uint8
    data = np.asarray(mem_data_deepcopy, dtype = bit_depth)
    return data

def read_h5(file_path: str) -> np.ndarray:
    if(bit_depth == "uint16"):
        bit_depth = np.uint16
    elif(bit_depth == "uint8"):
        bit_depth = np.uint8
    with h5py.File(file_path, "r") as f:
        key = list(f.keys())[0]
        data = np.array(f[key], dtype = bit_depth)
    return data

def read_img(read_dirpath: str, files: list, shape: tuple, bit_depth: str) -> np.ndarray: # always return 4d array (num_frames = 1, height, width, num_channels)
    # check only one file
    if(len(files) != 1):
        raise Exception("Expected len of files var = 1, received len = ", len(files), ". Please change format accordingly or double check files var")
    file = files[0]
    file_path = os.path.join(read_dirpath, file)
    # read data
    data = imageio.imread(file_path) 
    # ensure data is 4d
    if (data.ndim == 2):
        data = np.expand_dims(data, axis=0) 
        data = np.expand_dims(data, axis=3) 
    elif (data.ndim == 3):
        data = np.expand_dims(data, axis=0) 
    # ensure data shape and bit_depth are correct
    if (data.dtype != bit_depth):
        raise Exception("Expected bit_depth = ", bit_depth, ", but received: ", data.dtype)
    if (data.shape != shape):
        raise Exception("Expected shape = ", shape, ", but received: ", data.shape)
    # return 4d data
    return data

def read_vid(read_dirpath: str, files: list, shape: tuple, bit_depth: str) -> np.ndarray:  # always return 4d array (num_frames, height, width, num_channels)
    # check only one file
    if(len(files) != 1):
        raise Exception("Expected len of files var = 1, received len = ", len(files), ". Please change format accordingly or double check files var")
    file = files[0]
    filename, ext = file.split(".")
    ext = "." + ext
    file_path = os.path.join(read_dirpath, file)
    # check only one file
    if ext in [".h5"]:
        data = read_h5(file_path)
    elif ext in [".dat"]:
        data = read_dat(file_path, shape)
    else:
        data = imageio.mimread(file_path, memtest = "16GB") 
        if(bit_depth == "uint16"):
            bit_depth = np.uint16
        elif(bit_depth == "uint8"):
            bit_depth = np.uint8
        data = np.asarray(data, dtype = bit_depth) 
    # ensure data is 4d
    if (data.ndim == 3):
        data = np.expand_dims(data, axis=3) 
    # ensure data shape and bit_depth are correct
    if (data.dtype != bit_depth):
        raise Exception("Expected bit_depth = ", bit_depth, ", but received: ", data.dtype)
    if (data.shape != shape):
        raise Exception("Expected shape = ", shape, ", but received: ", data.shape)
    # return 4d data
    return data

def read_vid_from_imgs(read_dirpath: str, files: list, shape: tuple, bit_depth: str) -> np.ndarray: # always return 4d array (num_frames, height, width, num_channels)
    # ensure number of files matches expected shape
    if (shape[0] != len(files)):
        raise Exception("Expected number of frames = ", shape[0], ", receieved number of images = ", len(files))
    # ensure files are sorted correctly, in increasing numerical order
    files.sort()
    files = sorted(files, key = len)
    # read data
    data = []
    for i in range(len(files)):
        data.append(read_img(read_dirpath, [files[i]], (1, shape[1], shape[2], shape[3]), bit_depth)) 
    data = np.concatenate(data, axis=0)
    # ensure data shape and bit_depth are correct
    if (data.dtype != bit_depth):
        raise Exception("Expected bit_depth = ", bit_depth, ", but received: ", data.dtype)
    if (data.shape != shape):
        raise Exception("Expected shape = ", shape, ", but received: ", data.shape)
    # return 4d data
    return data

def read_visual_data(read_dirpath: str, files: list, shape: tuple, format: str, bit_depth: str) -> np.ndarray:
    # perform read action
    if format in ["single_img"]: # single image
        data = read_img(read_dirpath, files, shape, bit_depth)
    elif format in ["multiple_imgs"]: # multiple images
        data = read_vid_from_imgs(read_dirpath, files, shape, bit_depth)
    elif format in ["vid"]: # video or multiple single images
        data = read_vid(read_dirpath, files, shape, bit_depth)
    else:
        raise Exception("Expected format as one of ('single_img', 'multiple_imgs', 'vid'), but received: ", format)
    # ensure data is 4d
    if (data.ndim != 4):
        raise Exception("Expected data dim = 4, but received: ", data.ndim)
    # return 4d data
    return data

############################################### read meta_data functions

def read_json(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

############################################### read unidimensional_signal_data functions

def read_npy(file_path: str, allow_pickle=False) -> dict:
    data = np.load(file_path, allow_pickle=allow_pickle)
    return data

############################################### read rf_data functions

def read_pkl(file_path: str) -> dict:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def read_rf_data(read_dirpath: str, files: list, radar_config: tuple) -> np.ndarray:
    # check only one file
    if(len(files) != 1):
        raise Exception("Expected len of files var = 1, received len = ", len(files), ". Please change format accordingly or double check files var")
    file = files[0]
    filename, ext = file.split(".")
    ext = "." + ext
    file_path = os.path.join(read_dirpath, file)
    # perform read action
    if ext in [".raw"]: # single image
        organizer = radar.Organizer(file_path, radar_config)
        data = organizer.organize()
        timestamps = organizer.timestamps
    elif ext in [".pkl"]: # multiple images
        data = read_pkl(file_path)
        timestamps = None
    else:
        raise Exception("Expected ext as one of ('.raw', '.pkl'), but received: ", ext)
    # ensure data is 4d
    if (data.ndim != 4):
        raise Exception("Expected data dim = 4, but received: ", data.ndim)
    # return 4d data
    return data, timestamps

############################################### read audio functions

#TODO
