#Abstract Base Class for all data

import abc
import time
import os
from datetime import datetime

class Data(abc.ABC):
    @abc.abstractmethod
    def __init__(self, read_dirpath: str, write_dirpath: str, read_files: list, write_files: list, bit_depth: str, modality: str):
        self.read_dirpath = None
        self.write_dirpath = None
        self.read_files = None
        self.write_files = None

        self.bit_depth  = bit_depth
        self.modality = modality
        self.fps = None

        self.data = None
        self.status = False
        self.saved_or_displayed = False

    def __del__(self) -> None:
        """[Deconstructor, mainly just calls self.release_sensor()]"""
        self.release_data()


    ######### reading (creating), writing (saving) and releasing data
    
    @abc.abstractmethod
    def read(self, read_path, read_filename, read_ext) -> bool:
        """[Main Logic for reading data from file]
        Args:
            read_path (string)
            read_filename (string)
            read_ext (string)
        Returns:
            bool: [true if data is read]
        """
        pass


    @abc.abstractmethod
    def write(self, write_path, write_filename, write_ext) -> bool:
        """[Routine for writing data to file]
        Args:
            write_path (string)
            write_filename (string)
            write_ext (string)
        Returns:
            bool: [true if data is written]
        """
        pass
    
    @abc.abstractmethod
    def check_status(self):
        """[Routine for releasing data resources, can only be called after 
            writing to file]
        Returns:
            bool: [true if done]
        """
        pass

    @abc.abstractmethod
    def release_data(self):
        """[Routine for releasing data resources, can only be called after 
            writing to file]
        Returns:
            bool: [true if done]
        """
        pass

    # ######### transforming data

    # def transform(self) -> bool:
    #     """[Routine for transforming data]
    #     Returns:
    #         bool: [true if done]
    #     """
    #     # self.crop()
    #     # self.resizeSpatial()
    #     # self.resizeTemporal()
    #     # self.normalize()
    #     pass















