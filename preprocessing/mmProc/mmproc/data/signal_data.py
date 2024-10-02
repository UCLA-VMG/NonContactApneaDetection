from fileinput import filename
# from importlib.metadata import files
from mmproc.data.data import Data
import numpy as np

import mmproc.utils.utils_read as read
import mmproc.utils.utils_write as write
import mmproc.utils.utils_display as display
import mmproc.utils.utils_transform as transform
# import mmproc.utils.utils_radar as radar

class SignalData(Data):
    def __init__(self, read_dirpath: str = "", write_dirpath: str = "", read_files: list = [], write_files: list = [], bit_depth: str = "", modality: str = "", fps: int = 0):
        super().__init__(read_dirpath = read_dirpath, write_dirpath = write_dirpath, read_files = read_files, write_files = write_files, bit_depth = bit_depth, modality = modality)
        self.read_dirpath = read_dirpath
        self.write_dirpath = write_dirpath
        self.read_files = read_files
        self.write_files = write_files

        self.bit_depth = bit_depth
        self.modality = modality
        self.fps_data = fps
        # read
        self.data = None
        self.interpolated = None
        self.time = None
        self.sensor_stamps = None
        self.pc_stamps = None
        self.unroll_flag = None
        self.shape = None
        self.status = False
        self.offset = None

        # other
        if(read_files):
            self.read()      
    
    def __del__(self) -> None:
        self.release_data()
        # print("Released {} resources.".format(RFData))

    ######### reading (creating), writing (saving), displaying and releasing data
    
    def read(self) -> bool:
        self.data = read.read_signal_data(self.read_dirpath, self.read_files)
        return True

    def write(self) -> bool:
        self.status = write.write_signal_data(self.write_dirpath, self.write_files, self.data)
        self.check_status()
        return True

    def display(self) -> bool:
        self.status = display.display_signal_data()
        self.check_status()
        return True

    def check_status(self):
        if(not self.status):
            raise Exception("Could not write or display successfully!")
    
    def release_data(self):
        if(self.status):
            del self.data
        else:
            raise Exception("Attempting to delete data that has not been written or displayed!")

    ######### transforming data
    
    # def interpolate(self) -> bool:
    #     self.data = read.read_signal_data(self.read_dirpath, self.read_files)
    #     return True