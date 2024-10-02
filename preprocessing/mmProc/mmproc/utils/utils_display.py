import os
import shutil
import warnings
import tqdm
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt

import os
import shutil
import json
import copy
import pickle
from typing import Callable, List
from collections import MutableMapping
from contextlib import suppress
import warnings

import mmproc.utils.utils_read as read
import mmproc.utils.utils_dict as dict

############################################### Display functions 

############ visual_data

def display_single_img(data: np.ndarray, files: list) -> bool:
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
    # display img
    file = files[0]
    filename, ext = file.split(".")
    plt.figure()
    plt.title(filename)
    plt.imshow(data)
    plt.show(block=False)
    # close window
    user_input = None
    while user_input is None:
        user_input = input(f"Press any key to stop viewing this image: ")
        print()  
    plt.close()
    # return action status
    return True

def display_vid(data: np.ndarray, files: list, fps: int) -> bool:    
    # check only one file
    if(len(files) != 1):
        raise Exception("Expected len of files var = 1, received len = ", len(files), ". Please change format accordingly or double check files var")
    # display vid
    file = files[0]
    filename, ext = file.split(".")
    for i in range(data.shape[0]):
        cv2.imshow('Frame', data[i])
        cv2.waitKey(int(1/fps))
    # close window
    user_input = None
    while user_input is None:
        user_input = input(f"Press any key to stop viewing this video: ")
        print()  
    cv2.destroyAllWindows()
    # return action status
    return True

def display_visual_data(files: list, data: np.ndarray, format: str, kargs: dict) -> bool:
    # ensure data is 4d
    if (data.ndim != 4):
        raise Exception("Expected data dim = 4, but received: ", data.ndim)
    # perform display action
    if format in ["single_img"]: # single image
        result = display_single_img(data, files)
    elif format in ["vid", "multiple_imgs"]: # video or multiple single images
        fps = kargs["fps"]
        result = display_vid(data, files, fps)
    else:
        raise Exception("Expected format as one of ('single_img', 'multiple_imgs', 'vid'), but received: ", format)
    # return action status
    return result

############ meta_data

def print_meta_data(read_dirpath: str, write_dirpath: str, files: list):
    src_file_path, file_path = dict.get_meta_data_paths(read_dirpath, write_dirpath, files)
    data = read.read_json(src_file_path)
    print("metadata at src_file_path ", src_file_path, " : \n ", data )

############ signal_data

def display_signal(data, title: str = ""):
    plt.figure()
    plt.title(title)
    plt.plot(data)
    plt.show(block=False)
    user_input = None
    while user_input is None:
        user_input = input(f"Press any key to stop viewing this signal: ")
        print()  
    plt.close()

############ radar_data

def display_angle_range(angles: np.ndarray, ranges: np.ndarray, theta_bins: np.ndarray, frame_idx: int, convert: bool):
    if(convert):
        polar_grid = np.log((np.abs(theta_bins))[frame_idx].sum(0).T)
        # print(angles*180/np.pi)
        fig = plt.figure(figsize=[5,5])
        ax = fig.add_axes([0.1,0.1,0.8,0.8],polar=True)
        ax.set_xlim(-3.14159/2, 3.14159/2)
        ax.pcolormesh(angles,ranges,polar_grid,edgecolors='face')
        
        #ec='face' to avoid annoying gridding in pdf
        # plt.savefig('polar.png')
        # plt.imshow(, extent=[angles.min(), angles.max(), ranges.max(), ranges.min()])
        # plt.xlabel('Angle (Rad)')
        # plt.ylabel('Range (meters)')
        plt.title('Range and Angle - Framen Index: ', frame_idx)
        plt.show()
    else: 
        plt.imshow(np.log(np.fft.fftshift(np.abs(theta_bins), axes=2)[frame_idx].sum(0).T))
        plt.xlabel('Angle Bins')
        plt.ylabel('Range Bins')
        plt.title('Range and Angle - Framen Index: ', frame_idx)
        plt.show()

def display_velocity_range(velocities: np.ndarray, ranges: np.ndarray, range_doppler: np.ndarray, frame_idx: int, convert: bool):
    if(convert):
        plt.imshow((np.fft.fftshift(np.abs(range_doppler[frame_idx,::,::,::].sum(1)), axes=0).T), extent=[velocities.min(), velocities.max(), ranges.max(), ranges.min()])
        plt.xlabel('Velocity (m/s)')
        plt.ylabel('Range (m)')
        plt.title('Range and Velocity - Framen Index: ', frame_idx)
        plt.show()
    else: 
        plt.imshow((np.fft.fftshift(np.abs(range_doppler[frame_idx,::,::,::].sum(1)), axes=0).T))
        plt.xlabel('Velocity Bins')
        plt.ylabel('Range Bins')
        plt.title('Range and Velocity - Framen Index: ', frame_idx)
        plt.show()

def display_velocity_angle(velocities: np.ndarray, angles: np.ndarray, range_doppler: np.ndarray, frame_idx: int, convert: bool):
    if(convert):
        plt.imshow((np.fft.fftshift(np.fft.fftshift(np.abs(range_doppler[frame_idx,::,::,::].sum(2)), axes=0), axes=1)), extent=[velocities.min(), velocities.max(), angles.max(), angles.min()])
        plt.xlabel('Angle (Rad)')
        plt.ylabel('Velocity (m/s)')
        plt.title('Angle and Velocity - Framen Index: ', frame_idx)
        plt.show()
    else: 
        plt.imshow((np.fft.fftshift(np.fft.fftshift(np.abs(range_doppler[frame_idx,::,::,::].sum(2)), axes=0), axes=1)))
        plt.xlabel('Angle Bins')
        plt.ylabel('Velocity Bins')
        plt.title('Angle and Velocity - Framen Index: ', frame_idx)
        plt.show()

def display_rf_data(display_type: str, velocities: np.ndarray, angles: np.ndarray, ranges: np.ndarray, theta_bins: np.ndarray, range_doppler: np.ndarray, frame_idx: int, convert: bool):
    if(all(s in display_type for s in ["angle", "range"])):
        display_angle_range(angles, ranges, theta_bins, frame_idx, convert)
    elif(all(s in display_type for s in ["velocity", "range"])):
        display_velocity_range(velocities, ranges, range_doppler, frame_idx, convert)
    elif(all(s in display_type for s in ["velocity", "angle"])):
        display_velocity_angle(velocities, angles, range_doppler, frame_idx, convert)
    else:
        raise Exception("Expected display_type to include two of ('velocity', 'angle', 'range'), but received: ", display_type)

############ audio_data
