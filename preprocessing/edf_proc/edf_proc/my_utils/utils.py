import numpy as np
import os
import shutil
import tqdm
from tqdm import tqdm
from typing import Callable, List

import utils_file_organizer as file_organizer

from data.visual_data import VisualData

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

class FolderManager():
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        self.read_dataset_path = self.config["folder_manager"]["read_dataset_path"]
        self.write_dataset_path = self.config["folder_manager"]["write_dataset_path"]
        self.read_to_write = self.config["folder_manager"]["read_to_write"]

        self.read_dirpath = None
        self.write_dirpath = None
        self.dirs = None
        self.files = None

        self.read_dirpath_list = []
        self.write_dirpath_list = []
        self.dirs_list = []
        self.errors_list = []
        self.status = True
        self.edit_dict = None
        
    def __del__(self) -> None:
        pass
        # self.release_folders()
        # print("Released {} resources.".format(FolderManager))

    def edit_read_folders(self) -> list:
        # to be updated by user's own method
        pass

    def edit_write_folders(self) -> list:
        # to be updated by user's own method
        pass

    def traverse_folders_one_to_one(self,  edit_dict: dict = None):    
        for self.read_dirpath, self.dirs, self.files in tqdm(os.walk(self.read_dataset_path)):
            if (self.edit_dict is not None):
                edit_read_folders = edit_dict["visual_data"]["edit_read_folders"]
                edit_write_folders = edit_dict["visual_data"]["edit_write_folders"]
                # if user defines edit_read_folders, bind that to folder_manager object
                if not isinstance(edit_read_folders, type(None)):
                    bind(self, edit_read_folders)
                # if user defines edit_write_folders, bind that to folder_manager object
                if not isinstance(edit_write_folders, type(None)):
                    bind(self, edit_write_folders)
                # edit dirs to select folders
                self.edit_read_folders()
            
            # update lists
            self.read_dirpath_list.append(self.read_dirpath)
            self.dirs_list.append(self.dirs)  
            self.write_dirpath = self.read_dirpath.replace(self.read_dataset_path, self.write_dataset_path)
            if not os.path.exists(path=self.write_dirpath): # if write_dirpath DNE, create it
                os.makedirs(name=self.write_dirpath)
            self.write_dirpath_list.append(self.write_dirpath)

            # call fileManager to apply transformations
            if (not self.dirs): # if dirs is empty (i.e. deepest possible folder)
                # try:
                file_manager = file_organizer.FileManager(self.read_dirpath, self.write_dirpath, self.dirs, self.files, self.config, self.edit_dict)
                # begin file actions
                file_manager.file_action()
                del file_manager
                # except:
                    # self.errors_list.append(self.read_dirpath)
                    # print(f"Path {self.read_dirpath} has an error")

    def traverse_folders_one_to_many(self,  edit_dict: dict = None):
        for self.write_dirpath, self.dirs, self.files in tqdm(os.walk(self.read_dataset_path)):
            if (self.edit_dict is not None):
                edit_read_folders = edit_dict["visual_data"]["edit_read_folders"]
                edit_write_folders = edit_dict["visual_data"]["edit_write_folders"]
                # if user defines edit_read_folders, bind that to folder_manager object
                if not isinstance(edit_read_folders, type(None)):
                    bind(self, edit_read_folders)
                # if user defines edit_write_folders, bind that to folder_manager object
                if not isinstance(edit_write_folders, type(None)):
                    bind(self, edit_write_folders)
                # edit dirs to select folders
                self.edit_read_folders()
            
            # update lists
            self.write_dirpath_list.append(self.write_dirpath)
            self.dirs_list.append(self.dirs)  
            if not os.path.exists(path=self.write_dirpath): # if read_dirpath DNE, create it
                os.makedirs(name=self.write_dirpath)

            # call fileManager to apply transformations
            if (not self.dirs): # if dirs is empty (i.e. deepest possible folder)
                try:
                    file_manager = file_organizer.FileManager(self.read_dirpath, self.write_dirpath, self.dirs, self.files, self.config, self.edit_dict)
                    # begin file actions
                    file_manager.file_action()
                    del file_manager
                except:
                    self.errors_list.append(self.read_dirpath)
                    print(f"Path {self.read_dirpath} has an error")

    def traverse_folders_many_to_one(self,  edit_dict: dict = None):
        for self.read_dirpath, self.dirs, self.files in tqdm(os.walk(self.read_dataset_path)):
            if (self.edit_dict is not None):
                edit_read_folders = edit_dict["visual_data"]["edit_read_folders"]
                edit_write_folders = edit_dict["visual_data"]["edit_write_folders"]
                # if user defines edit_read_folders, bind that to folder_manager object
                if not isinstance(edit_read_folders, type(None)):
                    bind(self, edit_read_folders)
                # if user defines edit_write_folders, bind that to folder_manager object
                if not isinstance(edit_write_folders, type(None)):
                    bind(self, edit_write_folders)
                # edit dirs to select folders
                self.edit_read_folders()
            
            # update lists
            self.read_dirpath_list.append(self.read_dirpath)
            self.dirs_list.append(self.dirs)  
            write_dirpath = self.read_dirpath.replace(self.read_dataset_path, self.write_dataset_path)
            if not os.path.exists(path=write_dirpath): # if write_dirpath DNE, create it
                os.makedirs(name=write_dirpath)

            # call fileManager to apply transformations
            if (not self.dirs): # if dirs is empty (i.e. deepest possible folder)
                try:
                    file_manager = file_organizer.FileManager(self.read_dirpath, self.write_dirpath, self.dirs, self.files, self.config, self.edit_dict)
                    # begin file actions
                    file_manager.file_action()
                    del file_manager
                except:
                    self.errors_list.append(self.read_dirpath)
                    print(f"Path {self.read_dirpath} has an error")

    def process_data(self, edit_dict: dict = None):
        self.config.pop("folder_manager")
        self.edit_dict = edit_dict
        if (self.read_to_write == "one_to_one"):
            self.traverse_folders_one_to_one()
        if (self.read_to_write == "one_to_many"):
            self.traverse_folders_one_to_many()
        if (self.read_to_write == "many_to_one"):
            self.traverse_folders_many_to_one()


