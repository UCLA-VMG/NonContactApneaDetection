import numpy as np
import os
import shutil
import tqdm
from tqdm import tqdm
from typing import Callable, List

from data.visual_data import VisualData
# from mmproc.data.signal_data import SignalData
# from mmproc.data.audio_data import AudioData
# from mmproc.data.rf_data import RFData
# from mmproc.data.meta_data import MetaData

import utils_action_organizer as action_organizer

def bind(instance, func, as_name=None):
    """
    Bind the function *func* to *instance*, with either provided name *as_name*
    or the existing name of *func*. The provided *func* should accept the 
    instance as the first argument, i.e. "self".
    """
    if as_name is None:
        as_name = func.__name__
    bound_method = func.__get__(instance, instance.__class__)
    setattr(instance, as_name, bound_method)
    return bound_method

class FileManager():
    def __init__(self, read_dirpath: str, write_dirpath: str, dirs: list, files: list, config: dict, edit_dict: dict):
        super().__init__()
        self.read_dirpath = read_dirpath
        self.write_dirpath = write_dirpath
        self.dirs = dirs
        self.files = files
        self.config = config
        self.edit_dict = edit_dict

        self.read_files_dict = {}
        self.write_files_dict = {}

        self.num_frames_per_vid = None
        self.read_format = None
        self.read_ext = None
        self.read_filename = None
        self.write_format = None
        self.write_ext = None
        self.write_filename = None
        self.write_start_idx = None
        self.interpolate_format = None
        self.interpolate_ext = None
        self.write_vitals = None
        self.sensor_type = None

        self.config = self.config["file_manager"]
        
    def __del__(self) -> None:
        pass
        # self.release_folders()
        # print("Released {} resources.".format(FileManager))

    def edit_read_files(self) -> None:
        # to be updated by user's own method
        pass

    def edit_write_files(self) -> None:
        # to be updated by user's own method
        pass

    def file_action(self):
        # loop through data_types
        for data_type in (self.config.keys()): 
            if(self.edit_dict is None):
                self.build_edit_functions(data_type)
            else:
                edit_read_files = self.edit_dict[data_type]["edit_read_files"]
                edit_write_files = self.edit_dict[data_type]["edit_write_files"]
                print("edit_read_files: ", edit_read_files)

                print("Binding read!")
                bind(self, edit_read_files)
                self.edit_read_files()
                print("self.read_files_dict: ", self.read_files_dict)
                print("Binding write!")
                bind(self, edit_write_files)
                self.edit_write_files()

            data_type_config = self.config[data_type]
            init_config = data_type_config["read"]
            action_config = data_type_config

            if ("write" in data_type_config.keys()):
                # ensure read_files_dict and write_files_dict lengths are equal
                if (len(self.read_files_dict.keys()) != len(self.write_files_dict.keys())):
                    raise Exception("self.read_files_dict len = ", len(self.read_files_dict.keys()), "to match self.write_files_dict len = ", len(self.write_files_dict.keys()))
            # loop over read_files_dict
            for i in tqdm(range(len(self.read_files_dict.keys()))):
                read_files = self.read_files_dict[i]
                if ("write" in data_type_config.keys()):
                    write_files = self.write_files_dict[i]
                else:
                    write_files = []

                # apply action on read_files
                action_manager = action_organizer.ActionManager(self.read_dirpath, self.write_dirpath, read_files, write_files, data_type, init_config, action_config)
                action_manager.perform_actions()
                del action_manager

    ########################################################################################################## visual_data
    ########################################### e.g. convert visual_data format: multiple_imgs to vid
    # define read_visual_folders here if required

    # define read_visual_files here if required, must update self.read_files_dict
    def read_visual_files_to_file(self) -> None:
        # keep files with .bmp only
        read_files = [f for f in self.files if self.read_ext in f]
        # sort read_files
        read_files.sort()
        read_files = sorted(read_files, key = len)
        # find number of videos
        if (len(read_files) % self.num_frames_per_vid == 0):
            num_videos = int (len(read_files) / self.num_frames_per_vid)
        else:
            num_videos = int (len(read_files) / self.num_frames_per_vid) + 1
        # update self.files_dict
        self.read_files_dict = {}
        for i in range(num_videos):
            read_vid_files = read_files[ (i*self.num_frames_per_vid): (i*self.num_frames_per_vid) + self.num_frames_per_vid ]
            self.read_files_dict[i] = read_vid_files

    # define write_visual_folders here if required

    # define write_visual_files here if required, must update self.write_files_dict
    def write_visual_files_to_file(self) -> None:
        # keep files with .bmp only
        read_files = [f for f in self.files if self.read_ext in f]
        # sort read_files
        read_files.sort()
        read_files = sorted(read_files, key = len)
        # find number of videos
        if (len(read_files) % self.num_frames_per_vid == 0):
            num_videos = int (len(read_files) / self.num_frames_per_vid)
        else:
            num_videos = int (len(read_files) / self.num_frames_per_vid) + 1
        # update self.write_files
        self.write_files_dict = {}
        for i in range (num_videos):
            idx = self.write_start_idx + i
            self.write_files_dict[i] = [self.write_filename + str(idx) + self.write_ext]
        # ensure number of read videos and number of write videos is equal
        if (len(self.read_files_dict.keys()) != len(self.write_files_dict.keys()) ):
            raise Exception("Expected number of videos to write = ", len(self.read_files_dict.keys()), ", received number of videos to write = ", len(self.write_files_dict.keys()))

    ########################################### e.g. convert visual_data format: vid to multiple_imgs
    # define read_visual_folders here if required

    # define read_visual_files here if required, must update self.read_files_dict
    def read_visual_file_to_files(self) -> None:
        # keep files with .tiff only
        if (self.read_ext is not None):
            read_files = [f for f in self.files if self.read_ext in f]
        else:
            read_files = [f for f in self.files]
        # sort read_files
        read_files.sort()
        read_files = sorted(read_files, key = len)
        # find total num of videos
        self.read_files_dict = {}
        for i in range(len(read_files)):
            read_file = read_files[i]
            self.read_files_dict[i] = [read_file]

    # define write_visual_folders here if required

    # define write_visual_files here if required, must update self.write_files_dict
    def write_visual_file_to_files(self) -> None:
        # keep files with .tiff only
        if (self.read_ext is not None):
            read_files = [f for f in self.files if self.read_ext in f]
        else:
            read_files = [f for f in self.files]
        # sort read_files
        read_files.sort()
        read_files = sorted(read_files, key = len)
        # find total num of imgs
        num_read_files = len(read_files) # num of videos
        num_write_files = num_read_files * self.num_frames_per_vid
        write_files = []
        for i in range (num_write_files):
            idx = self.write_start_idx + i
            write_files.append(self.write_filename + str(idx) + self.write_ext)
        # update self.write_files
        self.write_files_dict = {}
        for i in range(num_read_files):
            write_vid_files = write_files[ (i*self.num_frames_per_vid): (i*self.num_frames_per_vid) + self.num_frames_per_vid ]
            self.write_files_dict[i] = write_vid_files

    ########################################### e.g. convert visual_data format: single_img to single_img
    # define read_visual_folders here if required

    # define read_visual_files here if required, must update self.read_files_dict
    def read_visual_file_to_file(self) -> None:
        # keep files with .bmp only
        if (self.read_ext is not None):
            read_files = [f for f in self.files if self.read_ext in f]
        else:
            read_files = [f for f in self.files]
        # sort read_files
        read_files.sort()
        read_files = sorted(read_files, key = len)
        # to be updated by user's own method
        for i in range(len(read_files)):
            read_file = read_files[i]
            self.read_files_dict[i] = [read_file]

    # define write_visual_folders here if required

    # define write_visual_files here if required, must update self.write_files_dict
    def write_visual_file_to_file(self) -> None:
        if (self.write_start_idx is not None):
            # keep files with .bmp only
            if (self.read_ext is not None):
                read_files = [f for f in self.files if self.read_ext in f]
            else:
                read_files = [f for f in self.files]
            # sort read_files
            read_files.sort()
            read_files = sorted(read_files, key = len)
            # keep files with .bmp only
            if (self.read_ext is not None) and (self.write_ext is not None):
                write_files = [f.replace(self.read_ext, self.write_ext) for f in read_files]
            else:
                write_files = read_files
            # sort write_files
            write_files.sort()
            write_files = sorted(write_files, key = len)
            for idx, f in enumerate(write_files):
                frame_idx = ''.join(ch for ch in f if ch.isdigit())
                f = f.replace(frame_idx, str(idx + self.write_start_idx))
                write_files[idx] = f
            # to be updated by user's own method
            for i in range(len(write_files)):
                write_file = write_files[i]
                self.write_files_dict[i] = [write_file]
        else:
            # keep files with .bmp only
            if (self.read_ext is not None):
                read_files = [f for f in self.files if self.read_ext in f]
            else:
                read_files = [f for f in self.files]
            # sort read_files
            read_files.sort()
            read_files = sorted(read_files, key = len)
            # keep files with .bmp only
            if (self.read_ext is not None) and (self.write_ext is not None):
                write_files = [f.replace(self.read_ext, self.write_ext) for f in read_files]
            else:
                write_files = read_files
            # sort write_files
            write_files.sort()
            write_files = sorted(write_files, key = len)
            # to be updated by user's own method
            for i in range(len(write_files)):
                write_file = write_files[i]
                self.write_files_dict[i] = [write_file]

    ########################################################################################################## vitals_data
    ### for reading raw vital data 
    # define read_vitals_files here if required, must update self.read_files_dict
    def read_vitals_files(self) -> None:
        # keep files with .bmp only
        read_files = [f for f in self.files if self.read_ext in f]
        # sort read_files
        read_files.sort()
        read_files = sorted(read_files, key = len)
        # update self.files_dict
        self.read_files_dict = {}
        self.read_files_dict[0] = read_files
        
    ### for writing vital dict     
    # define write_vitals_file here if required, must update self.write_files_dict
    def write_vitals_file(self) -> None:
        self.write_files_dict = {}
        self.write_files_dict[0] = [self.write_filename + self.write_ext]
    
    ### for reading processed vital data 
    # define read_vitals_file here if required, must update self.read_files_dict
    def read_vitals_file(self) -> None:
        self.read_files_dict = {}
        self.read_files_dict[0] = [self.read_filename + self.read_ext]
        
    ### for writing vital dict       
    # define write_vitals_files here if required, must update self.write_files_dict
    def write_vitals_files(self) -> None:
        write_files = []
        if (self.sensor_type in ["Philips", "philips", "MX800", "mx800", "Intellivue", "intellivue"]):
            if not self.write_vitals:
                self.write_vitals = ["NOM_PLETHWaveExport", "NOM_RESPWaveExport", "NOM_ECG_ELEC_POTL_AVFWaveExport", "NOM_ECG_ELEC_POTL_IIWaveExport", "NOM_ECG_ELEC_POTL_MCLWaveExport", "NOM_ECG_ELEC_POTL_VWaveExport", "NOM_TEMP", "NOM_ECG_CARD_BEAT_RATE", "NOM_ECG_AMPL_ST_I", "NOM_ECG_AMPL_ST_II", "NOM_ECG_AMPL_ST_III", "NOM_ECG_AMPL_ST_AVR", "NOM_ECG_AMPL_ST_AVL", "NOM_ECG_AMPL_ST_AVF", "NOM_ECG_AMPL_ST_V", "NOM_ECG_AMPL_ST_MCL", "NOM_PULS_OXIM_SAT_O2", "NOM_PLETH_PULS_RATE", "NOM_PULS_OXIM_PERF_REL", "NOM_PRESS_BLD_NONINV_SYS", "NOM_PRESS_BLD_NONINV_DIA", "NOM_PRESS_BLD_NONINV_MEA"]
            for vital in self.write_vitals:
                write_files.append(vital + self.write_ext)
        elif (self.sensor_type in ["Nihon_Kohden", "nihon_kohden", "nk"]):
            if not self.write_vitals:
                self.write_vitals = ["CHEST", "ABD", "LOC", "ROC", "EKG", "SpO2", "PTAF", "O2-M1", "C2-M1", "F4-M1", "R-R", "CHIN", "L LEG", "R LEG", "IMAGING"]
            for vital in self.write_vitals:
                write_files.append(vital + self.write_ext)
        # update self.files_dict
        self.write_files_dict = {}
        self.write_files_dict[0] = write_files

    ########################################################################################################## visual_data

    def build_edit_functions(self, data_type: str) -> dict:
        if (data_type == "visual_data"):
            self.read_format = self.config[data_type]["read"]["format"]
            self.read_ext = self.config[data_type]["read"]["ext"]
            self.write_format = self.config[data_type]["write"]["format"]
            self.write_ext = self.config[data_type]["write"]["ext"]
            if ( (all([s == "vid" for s in [self.read_format, self.write_format] ])) or (all([s == "single_img" for s in [self.read_format, self.write_format] ])) ):
                if ("start_idx" in (self.config[data_type]["write"].keys())):
                    self.write_start_idx =  self.config[data_type]["write"]["start_idx"]
                self.read_visual_file_to_file()
                self.write_visual_file_to_file()
            elif (self.read_format in ["vid"]) and (self.write_format in ["multiple_imgs"]):
                self.write_filename = self.config[data_type]["write"]["filename"]
                self.write_start_idx =  self.config[data_type]["write"]["start_idx"]
                self.num_frames_per_vid = self.config[data_type]["read"]["shape"][0]
                self.read_visual_file_to_files()
                self.write_visual_file_to_files()
            elif (self.read_format in ["multiple_imgs"]) and (self.write_format in ["vid"]):
                self.write_filename = self.config[data_type]["write"]["filename"]
                self.write_start_idx =  self.config[data_type]["write"]["start_idx"]
                self.num_frames_per_vid = self.config[data_type]["read"]["shape"][0]
                self.read_visual_files_to_file()
                self.write_visual_files_to_file()
            else:
                raise Exception("read_format and write_format do not match!")
        
        elif (data_type == "rf_data"):
            self.read_format = self.config[data_type]["read"]["format"]
            self.read_ext = self.config[data_type]["read"]["ext"]
            self.write_format = self.config[data_type]["write"]["format"]
            self.write_ext = self.config[data_type]["write"]["ext"]
            if (all([s == "single_file" for s in [self.read_format, self.write_format] ])):
                self.read_visual_file_to_file()
                self.write_visual_file_to_file()
            else:
                raise Exception("read_format and write_format do not match!")
        
        elif (data_type == "vitals_data"):
            self.read_format = self.config[data_type]["read"]["format"]
            self.read_ext = self.config[data_type]["read"]["ext"]
            
            if "interpolate" in self.config[data_type].keys():
                if "filename" in self.config[data_type]["interpolate"].keys() and "ext" in self.config[data_type]["interpolate"].keys():
                    self.interpolate_format = self.config[data_type]["interpolate"]["filename"]
                    self.interpolate_ext = self.config[data_type]["interpolate"]["ext"]
                else:
                    raise Exception("Missing one of 'filename' or 'ext' in interpolate key for processing vitals_data!")
            
            if (self.read_format in ["raw"]):
                self.read_vitals_files()
            elif (self.read_format in ["proc", "processed"]):
                self.read_vitals_file()
            
            if("write" in self.config[data_type].keys() ):
                self.write_format = self.config[data_type]["write"]["format"]
                self.write_ext = self.config[data_type]["write"]["ext"]
                if (self.write_format in ["dict"]):
                    self.write_filename = self.config[data_type]["write"]["filename"]
                    self.write_vitals_file()
                elif (self.write_format in ["img", "imgs", "array", "arrays"]):
                    self.write_vitals = self.config[data_type]["write"]["vitals"]
                    self.sensor_type = self.config[data_type]["read"]["sensor_type"]
                    self.write_vitals_files()

        elif (data_type == "file"):
            self.read_format = self.config[data_type]["read"]["format"]
            self.write_format = self.config[data_type]["write"]["format"]
            if (all([s == "single" for s in [self.read_format, self.write_format] ])):
                if ("start_idx" in (self.config[data_type]["write"].keys())):
                    self.write_start_idx =  self.config[data_type]["write"]["start_idx"]
                if ("ext" in (self.config[data_type]["read"].keys())):
                    self.read_ext =  self.config[data_type]["read"]["ext"]
                if ("ext" in (self.config[data_type]["write"].keys())):
                    self.write_ext =  self.config[data_type]["write"]["ext"]
                self.read_visual_file_to_file()
                self.write_visual_file_to_file()
            else:
                raise Exception("read_format and write_format do not match!")

                