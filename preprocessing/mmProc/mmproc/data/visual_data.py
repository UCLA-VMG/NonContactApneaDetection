from fileinput import filename
# from importlib.metadata import files
from mmproc.data.data import Data

import mmproc.utils.utils_read as read
import mmproc.utils.utils_write as write
import mmproc.utils.utils_display as display
import mmproc.utils.utils_transform as transform

class VisualData(Data):
    def __init__(self, read_dirpath: str, write_dirpath: str, read_files: list, write_files: list, format: str, shape: tuple, bit_depth: str, modality: str, fps: int):
        super().__init__(read_dirpath = read_dirpath, write_dirpath = write_dirpath, read_files = read_files, write_files = write_files, bit_depth = bit_depth, modality = modality)
        self.read_dirpath = read_dirpath
        self.write_dirpath = write_dirpath
        self.read_files = read_files
        self.write_files = write_files

        self.format = format
        self.shape = shape
        self.bit_depth = bit_depth
        self.modality = modality
        self.fps = fps

        self.data = None
        self.status = False

        # self.compression = config.getint("nir", "compression")
        self.kargs = {'fps': self.fps}
        self.read()

    def __del__(self) -> None:
        self.release_data()
        # print("Released {} resources.".format(VisualData))

    ######### reading (creating), writing (saving), displaying and releasing data
    
    def read(self) -> bool:
        self.data = read.read_visual_data(self.read_dirpath, self.read_files, self.shape, self.format, self.bit_depth)
        self.shape = self.data.shape
        return True

    def write(self, format: str, kargs: dict) -> bool:
        self.status = write.write_visual_data(self.write_dirpath, self.write_files, self.data, format, self.modality, kargs)
        self.check_status()
        return True

    def display(self) -> bool:
        self.status = display.display_visual_data(self.read_files, self.data, self.format, self.kargs)
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

    def crop(self, crop_coords, buffer) -> bool:
        """[Routine for cropping data]
        Returns:
            bool: [true if done]
        """
        self.data = transform.crop_visual_data(self.data, crop_coords, buffer, self.modality)
        self.shape = self.data.shape
        return True

    def resize_spatial(self, resize_dims) -> bool:
        """[Routine for resizing data spatially]
        Returns:
            bool: [true if done]
        """
        self.data = transform.resize_visual_data_spatial(self.data, self.bit_depth, self.format, resize_dims)
        self.shape = self.data.shape
        return True

    def resize_temporal(self, modulus) -> bool:
        """[Routine for resizing data temporally]
        Returns:
            bool: [true if done]
        """
        self.data = transform.resize_visual_data_temporal(self.data, self.bit_depth, self.format, modulus)
        self.shape = self.data.shape
        return True

    def normalize(self) -> bool:
        """[Routine for normalizing data]
        Returns:
            bool: [true if done]
        """
        self.data = transform.normalize_visual_data(self.data, self.bit_depth, self.format)
        self.shape = self.data.shape
        return True
    
    def bin(self, bit_depth, step_size) -> bool:
        """[Routine for normalizing data]
        Returns:
            bool: [true if done]
        """
        self.bit_depth = bit_depth
        self.data = transform.bin_visual_data(self.data, self.bit_depth, self.format, step_size)
        self.shape = self.data.shape
        return True
    
    def demosaic_bin(self, bit_depth, step_size) -> bool:
        """[Routine for normalizing data]
        Returns:
            bool: [true if done]
        """
        self.bit_depth = bit_depth
        self.data = transform.demosaic_bin_visual_data(self.data, self.bit_depth, self.format, step_size)
        self.shape = self.data.shape
        return True


