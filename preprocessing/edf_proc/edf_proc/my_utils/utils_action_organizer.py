import numpy as np
import os
import shutil
from typing import Callable, List

from data.visual_data import VisualData
from data.vitals_data import VitalsData
from data.rf_data import RFData

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

class ActionManager():
    def __init__(self, read_dirpath: str, write_dirpath: str, read_files: list, write_files: list, data_type: str, init_config: str, action_config: str):
        super().__init__()
        self.read_dirpath = read_dirpath
        self.write_dirpath = write_dirpath
        self.read_files = read_files
        self.write_files = write_files

        self.data_type = data_type
        self.init_config = init_config
        self.action_config = action_config
        self.status = False
        
    def __del__(self) -> None:
        pass
        # self.release_folders()
        # print("Released {} resources.".format(FileManager))

    def check_status(self) -> bool:
        if(not self.status):
            raise Exception("Could not complete action successfully!")

        ############################################### Copy File Functions

    def copy_file(self) -> bool:
        # check only one file
        if(len(self.read_files) != 1):
            raise Exception("Expected len of self.read_files var = 1, received len = ", len(self.read_files), ". Please change format accordingly or double check self.read_files var")
        # copy single file
        file = self.read_files[0]
        read_file_path = os.path.join(self.read_dirpath, file)
        if os.path.isfile(read_file_path):
            write_file_path = os.path.join(self.write_dirpath, file)
            shutil.copy(read_file_path, write_file_path)
        # return action status
        return True

    def copy_multiple_files(self) -> bool:
        # check more than one file
        if(len(self.read_files) <= 1):
            raise Exception("Expected len of self.read_files var > 1, received len = ", len(self.read_files), ". Please change format accordingly or double check self.read_files var")
        # copy multiple self.read_files
        for file in self.read_files:
            read_file_path = os.path.join(self.read_dirpath, file)
            if os.path.isfile(read_file_path):
                write_file_path = os.path.join(self.write_dirpath, file)
                shutil.copy(read_file_path, write_file_path)
        # return action status
        return True

    ############################################### Delete File Functions

    def delete_file(self) -> bool:
        # check only one file
        if(len(self.read_files) != 1):
            raise Exception("Expected len of self.read_files var = 1, received len = ", len(self.read_files), ". Please change format accordingly or double check self.read_files var")
        # delete single file
        file = self.read_files[0]
        read_file_path = os.path.join(self.read_dirpath, file)
        if os.path.isfile(read_file_path):
            # warnings.warn(f"Input File {read_file_path} will now be deleted.")
            os.remove(read_file_path)
        # return action status
        return True

    def delete_multiple_files(self) -> bool:
        # check more than one file
        if(len(self.read_files) <= 1):
            raise Exception("Expected len of self.read_files var > 1, received len = ", len(self.read_files), ". Please change format accordingly or double check self.read_files var")
        # delete multiple self.read_files
        for file in self.read_files:
            read_file_path = os.path.join(self.read_dirpath, file)
            if os.path.isfile(read_file_path):
                # warnings.warn(f"Input File {read_file_path} will now be deleted.")
                os.remove(read_file_path)
        # return action status
        return True
    
    ############################################### Delete File Functions

    def rename_file(self) -> bool:
        # check only one file
        if(len(self.read_files) != 1):
            raise Exception("Expected len of self.read_files var = 1, received len = ", len(self.read_files), ". Please change format accordingly or double check self.read_files var")
        if(len(self.write_files) != 1):
            raise Exception("Expected len of self.write_files var = 1, received len = ", len(self.write_files), ". Please change format accordingly or double check self.read_files var")
        # delete single file
        read_file = self.read_files[0]
        read_file_path = os.path.join(self.read_dirpath, read_file)
        rename_file = self.write_files[0]
        rename_file_path = os.path.join(self.read_dirpath, rename_file)
        if os.path.isfile(read_file_path):
            # warnings.warn(f"Input File {read_file_path} will now be deleted.")
            os.rename(read_file_path, rename_file_path)
        # return action status
        return True

    def rename_multiple_files(self) -> bool:
        # check more than one file
        if(len(self.read_files) <= 1):
            raise Exception("Expected len of self.read_files var > 1, received len = ", len(self.read_files), ". Please change format accordingly or double check self.read_files var")
        if(len(self.write_files) <= 1):
            raise Exception("Expected len of self.write_files var > 1, received len = ", len(self.write_files), ". Please change format accordingly or double check self.write_files var")
        if(len(self.read_files) != len(self.write_files)):
            raise Exception("Expected len of self.read_files = len of self.write_files. Instead received len of self.read_files = ", len(self.read_files), "len of self.read_files = ", len(self.write_files))
        # delete multiple self.read_files
        for idx, read_file in enumerate(self.read_files):
            read_file_path = os.path.join(self.read_dirpath, read_file)
            rename_file = self.write_files[idx]
            rename_file_path = os.path.join(self.read_dirpath, rename_file)
            if os.path.isfile(read_file_path):
                # warnings.warn(f"Input File {read_file_path} will now be deleted.")
                os.rename(read_file_path, rename_file_path)
        # return action status
        return True

    def perform_actions(self) -> bool:
        if (self.data_type == "visual_data"):
            visual_data = VisualData(self.read_dirpath, self.write_dirpath, self.read_files, self.write_files, self.init_config["format"], self.init_config["shape"], self.init_config["bit_depth"], self.init_config["modality"], self.init_config["fps"])
            # loop through actions
            for action in (self.action_config.keys()):
                if (action == "write"):
                    format = self.action_config[action]["format"]
                    kargs = {"fps": self.action_config[action]["fps"]}
                    self.status = visual_data.write(format, kargs)
                    self.check_status()
                elif (action == "display"):
                    self.status = visual_data.display()
                    self.check_status()
                elif (action == "crop"):
                    if ("crop_coords" in self.action_config[action].keys()):
                        crop_coords = self.action_config[action]["crop_coords"]
                    else:
                        crop_coords = ()
                    buffer = self.action_config[action]["buffer"]
                    self.status = visual_data.crop(crop_coords, buffer)
                    self.check_status()
                elif (action == "resize_spatial"):
                    resize_dims = self.action_config[action]["resize_dims"]
                    self.status = visual_data.resize_spatial(resize_dims)
                    self.check_status()
                elif (action == "bin"):
                    bit_depth = self.action_config[action]["bit_depth"]
                    step_size = self.action_config[action]["step_size"]
                    self.status = visual_data.bin(bit_depth, step_size)
                    self.check_status()
                elif (action == "demosaic_bin"):
                    bit_depth = self.action_config[action]["bit_depth"]
                    step_size = self.action_config[action]["step_size"]
                    self.status = visual_data.demosaic_bin(bit_depth, step_size)
                    self.check_status()
                elif (action == "resize_temporal"):
                    modulus = self.action_config[action]["modulus"]
                    self.status = visual_data.resize_temporal(modulus)
                    self.check_status()
                elif (action == "normalize"):
                    self.status = visual_data.normalize()
                    self.check_status()
                elif (action == "delete"):
                    if (visual_data.format in ["multiple_imgs"]):
                        self.status = self.delete_multiple_files()
                        self.check_status()
                    elif(visual_data.format in ["single_img", "vid"]):
                        self.status = self.delete_file()
                        self.check_status()
                elif (action == "copy"):
                    if (visual_data.format in ["multiple_imgs"]):
                        self.status = self.copy_multiple_files()
                        self.check_status()
                    elif(visual_data.format in ["single_img", "vid"]):
                        self.status = self.copy_file()
                        self.check_status()
                elif (action == "rename"):
                    if(visual_data.format in ["single_img", "vid"]):
                        self.status = self.rename_file()
                        self.check_status()
            del visual_data

        elif (self.data_type == "vitals_data"):
            vitals_data = VitalsData(self.read_dirpath, self.write_dirpath, self.read_files, self.write_files, self.init_config["format"], self.init_config["sensor_type"], self.init_config["offsets"], self.init_config["vitals"])
            # loop through actions
            for action in (self.action_config.keys()):
                if (action == "write"):
                    format = self.action_config[action]["format"]
                    write_attributes = self.action_config[action]["attributes"]
                    vitals_data.write(format, write_attributes)
                elif (action == "display"):
                    display_vitals = self.action_config[action]["vitals"]
                    display_attributes = self.action_config[action]["attributes"]
                    vitals_data.display(display_vitals, display_attributes)
                elif (action == "interpolate"):
                    ref_ts_filename = self.action_config[action]["filename"]
                    ref_ts_ext = self.action_config[action]["ext"]
                    ref_ts_file = [ref_ts_filename + ref_ts_ext]
                    vitals_data.interpolate(ref_ts_file)
                elif (action == "delete"):
                    if (vitals_data.format in ["raw"]):
                        self.status = self.delete_multiple_files()
                        self.check_status()
                    elif(vitals_data.format in ["proc", "processed"]):
                        self.status = self.delete_file()
                        self.check_status()
                elif (action == "copy"):
                    if (vitals_data.format in ["raw"]):
                        self.status = self.copy_multiple_files()
                        self.check_status()
                    elif(vitals_data.format in ["proc", "processed"]):
                        self.status = self.copy_file()
                        self.check_status()
                elif (action == "rename"):
                    if(visual_data.format in ["proc", "processed"]):
                        self.status = self.rename_file()
                        self.check_status()
            del vitals_data
        
        # elif (self.data_type == "signal_data"):
        #     signal_data = SignalData()
        #     # loop through actions
        #     for action in (self.action_config.keys()):
        #         if (action == "read"):
        #             signal_data.read()
        #         elif (action == "write"):
        #             signal_data.write()
        #         elif (action == "display"):
        #             signal_data.display()
        #         elif (action == "crop"):
        #             signal_data.crop()
        #         elif (action == "resize_spatial"):
        #             signal_data.resize_spatial()
        #         elif (action == "resize_temporal"):
        #             signal_data.resize_temporal()
        #         elif (action == "normalize"):
        #             signal_data.normalize()
        #         elif (action == "delete"):
        #             if (visual_data.format in ["multiple"]):
        #                 self.status = self.delete_multiple_files(self.read_dirpath, self.files)
        #                 self.check_status()
        #             elif(visual_data.format in ["single"]):
        #                 self.status = self.delete_file(self.read_dirpath, self.files)
        #                 self.check_status()
        #         elif (action == "copy"):
        #             if (visual_data.format in ["multiple"]):
        #                 self.status = self.copy_multiple_files(self.read_dirpath, self.write_dirpath, self.files)
        #                 self.check_status()
        #             elif(visual_data.format in ["single"]):
        #                 self.status = self.copy_file(self.read_dirpath, self.write_dirpath, self.files)
        #                 self.check_status()
        #     del signal_data

        # elif (self.data_type == "audio_data"):
        #     audio_data = AudioData()
        #     # loop through actions
        #     for action in (self.action_config.keys()):
        #         if (action == "read"):
        #             self.status = audio_data.read()
        #             self.check_status()
        #         elif (action == "write"):
        #             self.status = audio_data.write()
        #             self.check_status()
        #         elif (action == "display"):
        #             self.status = audio_data.display()
        #             self.check_status()
        #         elif (action == "crop"):
        #             self.status = audio_data.crop()
        #             self.check_status()
        #         elif (action == "resize_spatial"):
        #             self.status = audio_data.resize_spatial()
        #             self.check_status()
        #         elif (action == "resize_temporal"):
        #             self.status = audio_data.resize_temporal()
        #             self.check_status()
        #         elif (action == "normalize"):
        #             self.status = audio_data.normalize()
        #             self.check_status()
        #         elif (action == "delete"):
        #             if (visual_data.format in ["multiple"]):
        #                 self.status = self.delete_multiple_files(self.read_dirpath, self.files)
        #                 self.check_status()
        #             elif(visual_data.format in ["single"]):
        #                 self.status = self.delete_file(self.read_dirpath, self.files)
        #                 self.check_status()
        #         elif (action == "copy"):
        #             if (visual_data.format in ["multiple"]):
        #                 self.status = self.copy_multiple_files(self.read_dirpath, self.write_dirpath, self.files)
        #                 self.check_status()
        #             elif(visual_data.format in ["single"]):
        #                 self.status = self.copy_file(self.read_dirpath, self.write_dirpath, self.files)
        #                 self.check_status()
        #     del audio_data

        elif (self.data_type == "rf_data"):
            rf_data = RFData(self.read_dirpath, self.write_dirpath, self.read_files, self.write_files, self.init_config["bit_depth"], self.init_config["modality"], self.init_config["fps"], self.init_config["params"], self.init_config["velocity_params"], self.init_config["angle_granularity"])
            # loop through actions
            for action in (self.action_config.keys()):
                if (action == "read"):
                    self.status = rf_data.read()
                    self.check_status()
                elif (action == "write"):
                    self.status = rf_data.write()
                    self.check_status()
                elif (action == "display"):
                    display_type = self.action_config[action]["display_type"]
                    frame_idx = self.action_config[action]["frame_idx"]
                    convert = self.action_config[action]["convert"]
                    self.status = rf_data.display(display_type, frame_idx, convert)
                    self.check_status()
                elif (action == "delete"):
                    if (rf_data.format in ["multiple"]):
                        self.status = self.delete_multiple_files()
                        self.check_status()
                    elif(rf_data.format in ["single"]):
                        self.status = self.delete_file()
                        self.check_status()
                elif (action == "copy"):
                    if (rf_data.format in ["multiple"]):
                        self.status = self.copy_multiple_files()
                        self.check_status()
                    elif(rf_data.format in ["single"]):
                        self.status = self.copy_file()
                        self.check_status()
                elif (action == "rename"):
                    if(visual_data.format in ["single"]):
                        self.status = self.rename_file()
                        self.check_status()
            del rf_data

        # elif (self.data_type == "meta_data"):
        #     meta_data = MetaData()
        #     # loop through actions
        #     for action in (self.action_config.keys()):
        #         if (action == "read"):
        #             self.status = meta_data.read()
        #             self.check_status()
        #         elif (action == "write"):
        #             self.status = meta_data.write()
        #             self.check_status()
        #         elif (action == "display"):
        #             self.status = meta_data.display()
        #             self.check_status()
        #         elif (action == "crop"):
        #             self.status = meta_data.crop()
        #             self.check_status()
        #         elif (action == "resize_spatial"):
        #             self.status = meta_data.resize_spatial()
        #             self.check_status()
        #         elif (action == "resize_temporal"):
        #             self.status = meta_data.resize_temporal()
        #             self.check_status()
        #         elif (action == "normalize"):
        #             self.status = meta_data.normalize()
        #             self.check_status()
        #         elif (action == "delete"):
        #             if (visual_data.format in ["multiple"]):
        #                 self.status = self.delete_multiple_files()
        #                 self.check_status()
        #             elif(visual_data.format in ["single"]):
        #                 self.status = self.delete_file()
        #                 self.check_status()
        #         elif (action == "copy"):
        #             if (visual_data.format in ["multiple"]):
        #                 self.status = self.copy_multiple_files()
        #                 self.check_status()
        #             elif(visual_data.format in ["single"]):
        #                 self.status = self.copy_file()
        #                 self.check_status()
        #     del meta_data

        elif (self.data_type == "file"):
            for action in (self.action_config.keys()):
                if (action == "copy"):
                    self.status = self.copy_file()
                    self.check_status()
                elif (action == "copy_multiple_files"):
                    self.status = self.copy_multiple_files()
                    self.check_status()
                elif (action == "delete"):
                    self.status = self.delete_file()
                    self.check_status()
                elif (action == "delete_multiple_files"):
                    self.status = self.delete_multiple_files()
                    self.check_status()
                elif (action == "rename"):
                    print("Wil begin renaming now!")
                    self.status = self.rename_file()
                    self.check_status()
                elif (action == "rename_multiple_files"):
                    self.status = self.rename_multiple_files()
                    self.check_status()
        else:
            raise Exception("Expected data_type to be one of (visual_data, signal_data, audio_data, rf_data, metadata_data, file). Instead receieved: ", self.data_type)