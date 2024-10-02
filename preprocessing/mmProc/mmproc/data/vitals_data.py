from fileinput import filename
# from importlib.metadata import files
from mmproc.data.data import Data
import numpy as np

import mmproc.utils.utils_read as read
import mmproc.utils.utils_write as write
import mmproc.utils.utils_display as display
import mmproc.utils.utils_transform as transform
import mmproc.utils.utils_vitals as vitals
import mmproc.data.signal_data as signal_data

class VitalsData(Data):
    def __init__(self, read_dirpath: str, write_dirpath: str, read_files: list, write_files: list, format: str, sensor_type: str, offsets: dict, read_vitals: list):
        super().__init__(read_dirpath = read_dirpath, write_dirpath = write_dirpath, read_files = read_files, write_files = write_files, bit_depth = "", modality = "")
        self.read_dirpath = read_dirpath
        self.write_dirpath = write_dirpath
        self.read_files = read_files
        self.write_files = write_files

        self.format = format
        self.sensor_type = sensor_type
        self.ref_ts_file = None
        self.offsets = offsets
        self.read_vitals = read_vitals
        self.write_vitals = None
        # self.fps = fps
        # self.interpolated_fps = None
        # read
        self.write_attributes = None
        self.signals = {}
        self.status = False

        # other
        self.read()     
    
    def __del__(self) -> None:
        self.release_data()
        # print("Released {} resources.".format(RFData))

    ######### reading (creating), writing (saving), displaying and releasing data
    
    def read(self) -> bool:
        self.signals = vitals.read_vitals_data(self.read_dirpath, self.read_files, self.read_vitals, self.offsets, self.sensor_type, self.format)
        return True

    def write(self, format: str, write_attributes: list) -> bool:
        self.format = format
        self.write_attributes = write_attributes
        self.status = vitals.write_vitals_data(self.write_dirpath, self.write_files, self.signals, self.write_attributes, self.format)
        self.check_status()
        return True

    def display(self, display_vitals: list, display_attributes: list) -> bool:
        self.display_vitals = display_vitals
        self.display_attributes = display_attributes
        self.status = vitals.display_vitals_data(self.signals, self.display_vitals, self.display_attributes)
        self.check_status()
        return True

    def check_status(self):
        if(not self.status):
            raise Exception("Could not write or display successfully!")
    
    def release_data(self):
        if(self.status):
            for signal in self.signals.keys():
                self.signals[signal].status = True
            del self.data
        else:
            raise Exception("Attempting to delete data that has not been written or displayed!")

    ######### align vitals relative to some reference

    def interpolate(self, ref_ts_file: list) -> bool:
        # self.interpolated_fps = interpolated_fps
        self.ref_ts_file = ref_ts_file
        self.signals = vitals.interpolate_vitals_data(self.signals, self.read_dirpath, self.ref_ts_file, self.sensor_type)
        return True
    
    # def align_vitals_data(self) -> bool:
    #     self.signals = vitals.align_vitals_data(self.signals)
    #     return True
