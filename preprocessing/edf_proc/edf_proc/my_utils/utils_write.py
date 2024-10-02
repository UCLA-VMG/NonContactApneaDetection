from fileinput import filename
from ipaddress import v6_int_to_packed
import numpy as np
import os
import shutil
import json
import imageio
# import imageio_ffmpeg
import h5py
import pickle
from typing import Callable, List
from collections import MutableMapping
from contextlib import suppress
import warnings
import matplotlib.pyplot as plt

############################################### write visual_data functions

def write_img(write_dirpath: str, files: list, data: np.ndarray) -> bool:
    # check only one file
    if(len(files) != 1):
        raise Exception("Expected len of files var = 1, received len = ", len(files), ". Please change format accordingly or double check files var")
    # check only one frame
    if (data.ndim == 4) and (data.shape[0] != 1):
        raise Exception("Expected num of frames (of 4d data image) = 1, received num of frames = ", data.shape[0])
    # if 4d and only one frame, then squeeze into 3d image to allow write action
    if (data.ndim == 4) and (data.shape[0] == 1):
        data = np.squeeze(data, 0)
    # ensure data is 3d
    if (data.ndim != 3):
        raise Exception("Expected dim of data = 3, received dim = ", data.ndim)
    # write data
    file = files[0]
    filename, ext = file.split(".")
    ext = "." + ext
    file_path = os.path.join(write_dirpath, file)
    if ext in [".dat"]:
        write_dat(file_path, data)
    else:
        imageio.imwrite(file_path, data)
    # return action status
    return True

def write_h5(file_path: str, data: np.ndarray, modality: str) -> bool:
    with h5py.File(file_path, 'w') as f:
        f.create_dataset(modality, data = data)
    return True

def write_dat(file_path: str, data: np.ndarray) -> bool: # def write_dat(file_path: str, data: np.ndarray, bit_depth: str):
    fp = np.memmap(file_path, dtype=data.dtype, mode='w+', shape=data.shape)
    return True

def write_vid_as_imgs(write_dirpath: str, files: list, data: np.ndarray) -> bool:
    # ensure number of files matches expected shape
    if(len(files) != data.shape[0]):
        raise Exception("Expected len of files var = ", data.shape[0], ", received len = ", len(files), ". Please change format accordingly or double check files var")
    # write data
    for i in range(data.shape[0]):
        write_img(write_dirpath, [files[i]], data[i])
    # return action status
    return True

def write_vid(write_dirpath: str, files: list, data: np.ndarray, modality: str, kargs: dict) -> bool:
    # check only one file
    if(len(files) != 1):
        raise Exception("Expected len of files var = 1, received len = ", len(files), ". Please change format accordingly or double check files var")
    # write data
    file = files[0]
    filename, ext = file.split(".")
    ext = "." + ext
    file_path = os.path.join(write_dirpath, file)
    if ext in [".h5"]:
        write_h5(file_path, data, modality)
    elif ext in [".tiff"]: 
        imageio.mimwrite(file_path, data, bigtiff = True)
    else:
        imageio.mimwrite(file_path, data)
    # return action status
    return True  

def write_visual_data(write_dirpath: str, files: list, data: np.ndarray, format: str, modality: str, kargs: dict) -> bool:
    # ensure data is 4d
    if (data.ndim != 4):
        raise Exception("Expected data dim = 4, but received: ", data.ndim)
    # perform write action
    if format in ["single_img"]: # single image
        result = write_img(write_dirpath, files, data)
    elif format in ["multiple_imgs"]: # multiple images
        result = write_vid_as_imgs(write_dirpath, files, data)
    elif format in ["vid"]: # video or multiple single images
        result = write_vid(write_dirpath, files, data, modality, kargs)
    else:
        raise Exception("Expected format as one of ('single_img', 'multiple_imgs', 'vid'), but received: ", format)
    # return action status
    return result

############################################### write meta_data functions

def write_json(file_path: str, data: dict) -> bool:
    with open(file_path, "w") as outfile:
        json.dump(data, outfile, indent = 4)
    return True

def gen_meta_data(input_dict: dict, output_dict: dict, gen_dict: dict, edit_dict: dict, read_dirpath: list, write_dirpath: list, files: list) -> bool:
    output_file_format = output_dict["output_ext"]
    if output_file_format != ".json":
        raise Exception(f"Invalid Output File Format entered! Expected .json, instead detected {output_file_format}.")
    output_filename = output_dict["output_filename"]
    file_path = os.path.join(write_dirpath, output_filename + output_file_format)
    write_json(file_path, edit_dict)
    return True

############################################### write unidimensional_signal_data functions

def write_npy(file_path: str, data: np.ndarray) -> bool:
    np.save(file_path, data)
    return True

def write_signal_as_img(file_path: str, data: np.ndarray, title: str = "") -> bool:
    plt.figure()
    plt.plot(data)
    plt.title(title)
    plt.savefig(file_path)
    plt.close()
    return True

############################################### write rf_data functions

def write_pkl(file_path: str, data: np.ndarray) -> bool:
    with open(file_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    return True

def write_rf_data(write_dirpath: str, files: list, data: np.ndarray) -> bool:
    # check only one file
    if(len(files) != 1):
        raise Exception("Expected len of files var = 1, received len = ", len(files), ". Please change format accordingly or double check files var")
    # ensure data is 4d
    if (data.ndim != 4):
        raise Exception("Expected data dim = 4, but received: ", data.ndim)
    # write data
    file = files[0]
    filename, ext = file.split(".")
    ext = "." + ext
    file_path = os.path.join(write_dirpath, file)
    if ext in [".pkl"]:
        write_pkl(file_path, data)
    elif ext in [".dat"]: 
        write_dat(file_path, data)
    elif ext in [".npy"]: 
        write_npy(file_path, data)
    else:
        raise Exception("Expected ext as one of ('.pkl', '.dat', '.npy'), but received: ", ext)
    # return action status
    return True  

def write_txt(write_dirpath: str, files: list, data: list):
    # check only one file
    if(len(files) != 1):
        raise Exception("Expected len of files var = 1, received len = ", len(files), ". Please change format accordingly or double check files var")
    file = files[0]
    filename, ext = file.split(".")
    ext = "." + "txt"
    file = filename + ext
    data = sorted(set(data), key=data.index)
    file_path = os.path.join(write_dirpath, file)
    with open(file_path, 'w') as f:
        # print(data)
        data_tbw = '\n'.join([str(i) for i in data])
        # print(data_tbw)
        f.write(data_tbw)
    return True

############################################### write audio functions

#TODO
        