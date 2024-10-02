from typing import Callable
import yaml
import time

import mmproc.utils.utils as utils
import mmproc.config.config_parser as config_parser

# define edit_read_folders here if required

# define edit_read_files here if required, must update self.read_files_dict
def edit_read_files(self):
    # keep files with .bmp only
    read_files = [f for f in self.files if ".tiff" in f and "NIR_Camera_940_" in f]
    # sort read_files
    read_files.sort()
    read_files = sorted(read_files, key = len)
    # find number of videos
    num_frames = 43
    if (len(read_files) % num_frames == 0):
        num_videos = int (len(read_files) / num_frames)
    else:
        num_videos = int (len(read_files) / num_frames) + 1
    # update self.files_dict
    self.read_files_dict = {}
    for i in range(num_videos):
        read_vid_files = read_files[ (i*num_frames): (i*num_frames) + num_frames ]
        self.read_files_dict[i] = read_vid_files

# define edit_write_folders here if required

# define edit_write_files here if required, must update self.write_files_dict
def edit_write_files(self):
    # keep files with .bmp only
    read_files = [f for f in self.files if ".bmp" in f]
    # sort read_files
    read_files.sort()
    read_files = sorted(read_files, key = len)
    # find number of videos
    num_frames = 43
    if (len(read_files) % num_frames == 0):
        num_videos = int (len(read_files) / num_frames)
    else:
        num_videos = int (len(read_files) / num_frames) + 1
    # update self.write_files
    self.write_files_dict = {}
    for i in range (num_videos):
        self.write_files_dict[i] = ["NIR_Camera_" + str(i) + ".tiff"]
    # ensure number of read videos and number of write videos is equal
    if (len(self.read_files_dict.keys()) != len(self.write_files_dict.keys()) ):
        raise Exception("Expected number of videos to write = ", len(self.read_files_dict.keys()), ", received number of videos to write = ", len(self.write_files_dict.keys()))

# Two options:
# option #1: User defines custom edit_dict as below
# edit_dict = {}
# edit_dict["visual_data"]["edit_read_files"] = edit_read_files
# edit_dict["visual_data"]["edit_write_files"] = edit_write_files
# edit_dict["visual_data"]["edit_read_folders"] = None
# edit_dict["visual_data"]["edit_write_folders"] = None
# option #2: make edit_dict of type None
edit_dict = None


if __name__ == '__main__':
    with open(r"C:\Users\Adnan\Desktop\mmProc\mmproc\config\config.yml", "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)

    proc_dataset_methods = ["OSA_Thermal_crop_resize"]
    
    start_time = time.time()

    for proc_data in proc_dataset_methods:
        print("*"*50)
        print("Processing dataset via: ", proc_data)
        proc_config = config[proc_data]
        for proc_key in proc_config.keys():
            print("Beginning process: ", proc_key)
            curr_config = proc_config[proc_key]
            config_parser.check_config()

            folder_manager = utils.FolderManager(curr_config)
            folder_manager.process_data(edit_dict)

    end_time = time.time()
    print("Program Execution Time: ", end_time - start_time, " seconds.")