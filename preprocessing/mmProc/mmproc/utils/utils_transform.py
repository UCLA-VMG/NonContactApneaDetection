import os
import shutil
import warnings
import tqdm
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
import colour
import colour_demosaicing

import os
import shutil
import json
import copy
import pickle
from typing import Callable, List
from collections import MutableMapping
from contextlib import suppress
import warnings

import mmproc.utils.utils_facedet as facedet

############################################### Normalize functions 

def normalize_single_img(data: np.ndarray, bit_depth: str) -> np.ndarray:
    # check only one frame
    if (data.ndim == 4) and (data.shape[0] != 1):
        raise Exception("Expected num of frames (of 4d data image) = 1, received num of frames = ", data.shape[0])
    # if 4d and only one frame, then squeeze into 3d image to allow write action
    if (data.ndim == 4) and (data.shape[0] == 1):
        data = np.squeeze(data, 0)
    # ensure data is 3d
    if (data.ndim != 3):
        raise Exception("Expected dim of data = 3, received dim = ", data.ndim)
    # normalize img
    mask = np.ma.masked_invalid(data)
    data[mask.mask] = 0.0
    data = cv2.normalize(src=data, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    data[mask.mask] = 255.0
    # ensure data is 4d
    if (data.ndim == 2):
        data = np.expand_dims(data, axis=0) 
        data = np.expand_dims(data, axis=3) 
    elif (data.ndim == 3):
        data = np.expand_dims(data, axis=0) 
    # ensure data bit_depth is correct
    if(bit_depth == "uint16"):
        bit_depth = np.uint16
    elif(bit_depth == "uint8"):
        bit_depth = np.uint8
    if (data.dtype != bit_depth):
        raise Exception("Expected bit_depth = ", bit_depth, ", but received: ", data.dtype)
    # return data
    return data

def normalize_vid(data: np.ndarray, bit_depth: str, resize_dims: tuple):    
    # normalize vid
    for i in range(data.shape[0]):
        data[i] = normalize_single_img(data, bit_depth)[0]
    # return data
    return data

def normalize_visual_data(data: np.ndarray, bit_depth: str, format: str):    
    # ensure data is 4d
    if (data.ndim != 4):
        raise Exception("Expected data dim = 4, but received: ", data.ndim)
    # normalize data
    if format in ["single_img"]: # single image
        data = normalize_single_img(data, bit_depth)
    elif format in ["multiple_imgs", "vid"]: # multiple images
        data = normalize_vid(data, bit_depth)
    else:
        raise Exception("Expected format as one of ('single_img', 'multiple_imgs', 'vid'), but received: ", format)
    # ensure data is 4d
    if (data.ndim != 4):
        raise Exception("Expected data dim = 4, but received: ", data.ndim)
    # return data
    return data

############################################### Crop Data functions 

############ visual_data

def crop_detector(data: np.ndarray, buffer: tuple, modality: str) -> np.ndarray:
    if (modality == "thermal"):
        detector = facedet.ThermalFaceDetector(r"C:\Users\Adnan\Documents\Github\mmproc\config\thermal_face_automl_edge_fast.tflite")
    else:
        detector = facedet.FaceDetector()
    data = detector.crop_data(data, buffer)
    return data

def crop_manual(data, crop_coords = (574, 1148, 900, 900), buffer = (0.0, 0.0)):
    # get crop coords and buffer
    x, y, w, h = crop_coords
    x_buffer, y_buffer = buffer
    # update crop coords to apply buffer
    x = max(int(x - (w * x_buffer )), 0)
    y = max(int(y - (h * y_buffer)), 0)
    w = int(w * (1 + 2 * x_buffer ))
    h = int(h * (1 + 2 * y_buffer))
    # crop data
    data = data[:, y : y + h, x : x + w, :]
    # return data
    return data

def crop_visual_data(data: np.ndarray, crop_coords: tuple, buffer: tuple, modality: str) -> np.ndarray:
    # ensure data is 4d
    if (data.ndim != 4):
        raise Exception("Expected data dim = 4, but received: ", data.ndim)
    # perform crop action
    if (len(crop_coords) == 0): # crop_coords is empty, then apply face detection
        data = crop_detector(data, buffer, modality)
    elif (len(crop_coords) == 4): # multiple images
        data = crop_manual(data, crop_coords, buffer)
    else:
        raise Exception("Expected crop_coords tuple to be of length 0 (for nndl face detection crop) or length 4 (for manual crop using coords provided), but received length: ", len(crop_coords))
    # ensure data is 4d
    if (data.ndim != 4):
        raise Exception("Expected data dim = 4, but received: ", data.ndim)
    # return data
    return data

############ signal_data

#TODO

############ audio_data

#TODO

############ rf_data

#TODO

############################################### Resize Data functions 

############ visual_data

def resize_single_img_spatial(data: np.ndarray, bit_depth: str, resize_dims: tuple):
    # check only one frame
    if (data.ndim == 4) and (data.shape[0] != 1):
        raise Exception("Expected num of frames (of 4d data image) = 1, received num of frames = ", data.shape[0])
    # if 4d and only one frame, then squeeze into 3d image to allow write action
    if (data.ndim == 4) and (data.shape[0] == 1):
        data = np.squeeze(data, 0)
    # ensure data is 3d
    if (data.ndim != 3):
        raise Exception("Expected dim of data = 3, received dim = ", data.ndim)
    # resize img
    data = cv2.resize(data, resize_dims, cv2.INTER_LINEAR)
    # ensure data is 4d
    if (data.ndim == 2):
        data = np.expand_dims(data, axis=0) 
        data = np.expand_dims(data, axis=3) 
    elif (data.ndim == 3):
        data = np.expand_dims(data, axis=0) 
    # ensure data bit_depth is correct
    if(bit_depth == "uint16"):
        bit_depth = np.uint16
    elif(bit_depth == "uint8"):
        bit_depth = np.uint8
    if (data.dtype != bit_depth):
        raise Exception("Expected bit_depth = ", bit_depth, ", but received: ", data.dtype)
    # return data
    return data

def resize_vid_spatial(data: np.ndarray, bit_depth: str, resize_dims: tuple):    
    # resize vid
    resized_data = []
    for i in range(data.shape[0]):
        frame = data[i]
        frame = np.expand_dims(frame, axis=0) 
        frame = resize_single_img_spatial(frame, bit_depth, resize_dims)
        np.delete(data, data[i])
        resized_data.append(frame)
    resized_data = np.concatenate(resized_data, axis=0)
    # return data
    return resized_data

def resize_visual_data_spatial(data: np.ndarray, bit_depth: str, format: str, resize_dims: tuple):    
    # ensure data is 4d
    if (data.ndim != 4):
        raise Exception("Expected data dim = 4, but received: ", data.ndim)
    # ensure resize_dims len is 2 (for height and width)
    if (len(resize_dims) != 2): # if resize_dims does not contain 2 element, error out
        raise Exception("Expected resize_dims len = 2, but received: ", len(resize_dims))
    # resize data spatially
    if format in ["single_img"]: # single image
        data = resize_single_img_spatial(data, bit_depth, resize_dims)
    elif format in ["multiple_imgs", "vid"]: # multiple images
        data = resize_vid_spatial(data, bit_depth, resize_dims)
    else:
        raise Exception("Expected format as one of ('single_img', 'multiple_imgs', 'vid'), but received: ", format)
    # ensure data is 4d
    if (data.ndim != 4):
        raise Exception("Expected data dim = 4, but received: ", data.ndim)
    # return data
    return data

def resize_visual_data_temporal(data: np.ndarray, bit_depth: str, format: str, modulus: int):    
    # ensure data is 4d
    if (data.ndim != 4):
        raise Exception("Expected data dim = 4, but received: ", data.ndim)
    # resize vid temporally
    if format in ["multiple_imgs", "vid"]: # multiple images
        # find frame idxs to remove
        idx_range = np.arange(0, data.shape[0], 1)
        idx_keep = idx_range
        idx_keep = idx_keep[::modulus, :, :, :]
        idx_remove = np.delete(idx_range, idx_keep)
        # resize vid
        data = np.delete(data, idx_remove, axis = 0)
    else:
        raise Exception("Expected format as one of ('multiple_imgs', 'vid'), but received: ", format)
    # ensure data bit_depth is correct
    if(bit_depth == "uint16"):
        bit_depth = np.uint16
    elif(bit_depth == "uint8"):
        bit_depth = np.uint8
    if (data.dtype != bit_depth):
        raise Exception("Expected bit_depth = ", bit_depth, ", but received: ", data.dtype)
    # ensure data is 4d
    if (data.ndim != 4):
        raise Exception("Expected data dim = 4, but received: ", data.ndim)
    # return data
    return data


# def bin_image(img, size=2):
#     bin_imgs = []
#     for i in range(size):
#         for j in range(size):
#             bin_imgs.append(img[i::size,j::size,:])
#     binned_img = np.mean(np.array(bin_imgs), axis = 0)
#     return binned_img

# def demosaic_bin_image(img, size = 2):
#     img_demosaic = colour_demosaicing.demosaicing_CFA_Bayer_bilinear(img)
#     binned_img = bin_image(img_demosaic, size=size)
#     return binned_img

def bin_single_img(data: np.ndarray, bit_depth: str, step_size: int):
    # check only one frame
    if (data.ndim == 4) and (data.shape[0] != 1):
        raise Exception("Expected num of frames (of 4d data image) = 1, received num of frames = ", data.shape[0])
    # if 4d and only one frame, then squeeze into 3d image to allow write action
    if (data.ndim == 4) and (data.shape[0] == 1):
        data = np.squeeze(data, 0)
    # ensure data is 3d
    if (data.ndim != 3):
        raise Exception("Expected dim of data = 3, received dim = ", data.ndim)
    # bin img
    bin_list = []
    for i in range(step_size):
        for j in range(step_size):
            bin_list.append(data[i::step_size,j::step_size,:])
    
    
    data = np.mean(np.array(bin_list), axis = 0)
    # ensure data is 4d
    if (data.ndim == 2):
        data = np.expand_dims(data, axis=0) 
        data = np.expand_dims(data, axis=3) 
    elif (data.ndim == 3):
        data = np.expand_dims(data, axis=0) 
    # ensure data bit_depth is correct
    if(bit_depth == "uint16"):
        bit_depth = np.uint16
    elif(bit_depth == "uint8"):
        bit_depth = np.uint8
    if (data.dtype != bit_depth):
        raise Exception("Expected bit_depth = ", bit_depth, ", but received: ", data.dtype)
    # return data
    return data

def bin_vid(data: np.ndarray, bit_depth: str, step_size: int):    
    # resize vid
    resized_data = []
    for i in range(data.shape[0]):
        frame = data[i]
        frame = np.expand_dims(frame, axis=0) 
        frame = bin_single_img(frame, bit_depth, step_size)
        np.delete(data, data[i])
        resized_data.append(frame)
    resized_data = np.concatenate(resized_data, axis=0)
    # return data
    return resized_data

def bin_visual_data(data: np.ndarray, bit_depth: str, format: str, step_size: int):    
    # ensure data is 4d
    if (data.ndim != 4):
        raise Exception("Expected data dim = 4, but received: ", data.ndim)
    # # ensure resize_dims len is 2 (for height and width)
    # if (len(step_size) != 2): # if step_size does not contain 2 element, error out
    #     raise Exception("Expected step_size len = 2, but received: ", len(step_size))
    # resize data spatially
    if format in ["single_img"]: # single image
        data = bin_single_img(data, bit_depth, step_size)
    elif format in ["multiple_imgs", "vid"]: # multiple images
        data = bin_vid(data, bit_depth, step_size)
    else:
        raise Exception("Expected format as one of ('single_img', 'multiple_imgs', 'vid'), but received: ", format)
    # ensure data is 4d
    if (data.ndim != 4):
        raise Exception("Expected data dim = 4, but received: ", data.ndim)
    # return data
    return data

def demosaic_bin_single_img(data: np.ndarray, bit_depth: str, step_size: int):
    # check only one frame
    if (data.ndim == 4) and (data.shape[0] != 1):
        raise Exception("Expected num of frames (of 4d data image) = 1, received num of frames = ", data.shape[0])
    # if 4d and only one frame, then squeeze into 3d image to allow write action
    if (data.ndim == 4) and (data.shape[0] == 1):
        data = np.squeeze(data, 0)
    # ensure data is 3d
    if (data.ndim != 3):
        raise Exception("Expected dim of data = 3, received dim = ", data.ndim)
    # demosaic binned img
    if(data.ndim != 2 and data.ndim == 3):
        data = np.squeeze(data, 2)
    data = colour_demosaicing.demosaicing_CFA_Bayer_bilinear(data)
    data = bin_single_img(data, bit_depth, step_size)
    # ensure data is 4d
    if (data.ndim == 2):
        data = np.expand_dims(data, axis=0) 
        data = np.expand_dims(data, axis=3) 
    elif (data.ndim == 3):
        data = np.expand_dims(data, axis=0) 
    # ensure data bit_depth is correct
    if(bit_depth == "uint16"):
        bit_depth = np.uint16
    elif(bit_depth == "uint8"):
        bit_depth = np.uint8
    if (data.dtype != bit_depth):
        raise Exception("Expected bit_depth = ", bit_depth, ", but received: ", data.dtype)
    # return data
    return data

def demosaic_bin_vid(data: np.ndarray, bit_depth: str, step_size: int):    
    # resize vid
    resized_data = []
    for i in range(data.shape[0]):
        frame = data[i]
        frame = np.expand_dims(frame, axis=0) 
        frame = demosaic_bin_single_img(frame, bit_depth, step_size)
        np.delete(data, data[i])
        resized_data.append(frame)
    resized_data = np.concatenate(resized_data, axis=0)
    # return data
    return resized_data

def demosaic_bin_visual_data(data: np.ndarray, bit_depth: str, format: str, step_size: int):    
    # ensure data is 4d
    if (data.ndim != 4):
        raise Exception("Expected data dim = 4, but received: ", data.ndim)
    # # ensure resize_dims len is 2 (for height and width)
    # if (len(step_size) != 2): # if step_size does not contain 2 element, error out
    #     raise Exception("Expected step_size len = 2, but received: ", len(step_size))
    # resize data spatially
    if format in ["single_img"]: # single image
        data = demosaic_bin_single_img(data, bit_depth, step_size)
    elif format in ["multiple_imgs", "vid"]: # multiple images
        data = demosaic_bin_vid(data, bit_depth, step_size)
    else:
        raise Exception("Expected format as one of ('single_img', 'multiple_imgs', 'vid'), but received: ", format)
    # ensure data is 4d
    if (data.ndim != 4):
        raise Exception("Expected data dim = 4, but received: ", data.ndim)
    # return data
    return data

############ signal_data

#TODO

############ audio_data

#TODO

############ rf_data

#TODO