import numpy as np
import os
import pickle
import scipy.signal
from matplotlib import pyplot as plt
import my_utils.utils_write as write
import my_utils.utils_read as read
import my_utils.utils_transform as transform
from tqdm import tqdm

# radar_read: reads radar data from multiple folders/trials
# read_dir_path: path to folder containing each trial folder
def radar_read(read_dir_path):
    data = np.zeros([1,1])
    for folder in os.listdir(read_dir_path):
        folder_path = os.path.join(read_dir_path, folder)
        file_path = os.path.join(folder_path, "FMCW_Radar.npy")
        if data.shape == (1,1):
            data = np.load(file_path)
        else:
            data = np.concatenate((data, np.load(file_path)), axis = 0)
    
    return data

# split_radar: splits data from multiple trials into several small chunks
# read_dir_path: path to folder containing each trial folder
# write_dir_path: path to folder to contain subfolders with the chunks
# patient_number: string containing patient id number
def split_radar(read_dir_path, write_dir_path, num_samples, patient_number):
    if not os.path.exists(write_dir_path):
        os.mkdir(write_dir_path)
        
    data = radar_read(read_dir_path)
    for i in range(int(data.shape[0] / num_samples)): #for every minute in the data
        folder_path = os.path.join(write_dir_path, "v_" + patient_number + "_" + str(i))
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        file_path = os.path.join(folder_path, "FMCW_Radar.pkl")
        write.write_pkl(file_path, data[num_samples*i:num_samples*(i+1)])

    return

# list_filenames: get ordered list of all image files from multiple trials
# image names should have format <file_name>i<ext> (e.g. Thermal_Camera_0)
# folder_paths: list of paths to trial folders containing individual images 
# trial_length: number of individual images in each folder
# file_name: file name excluding index and extension (e.g. "Thermal_Camera_")
# ext: file extension
# start_idx: optional parameter for fixing things if you mess up
def list_filenames(folder_paths, trial_length, file_name, ext, start_idx = 0):
    
    file_paths = []
    for folder_path in folder_paths:
        for i in range(trial_length - start_idx):
            file_path = os.path.join(folder_path, file_name + str(i + start_idx) + ext)
            file_paths.append(file_path)
            

    return file_paths

# move_files: moves files from the ordered list file_paths in chunks of num_frames
#             into subfolders of write_dir_path
# write_dir_path: path to folder to contain subfolders with the chunks
# num_frames: number of frames to be placed in each chunk
# patient_number: string containing patient id number
# file_paths: ordered list of paths to each frame
# file_name: file name excluding index and extension (e.g. "Thermal_Camera_")
# ext: file extension
# start_idx: optional parameter for fixing things if you mess up
def move_files(write_dir_path, num_frames, patient_number, file_paths, file_name, ext, start_idx = 0):
    
    for i in range(int(len(file_paths) / num_frames)):
        folder_path = os.path.join(write_dir_path, "v_" + patient_number + "_" + str(int(start_idx / num_frames + i)))
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        for j in range(num_frames):
            read_file_path = file_paths[num_frames * i + j]
            write_file_path = os.path.join(folder_path, file_name + str(j) + ext)
            os.rename(read_file_path, write_file_path)
    
    return

# split_camera: moves images contained in folders folder_paths in chunks of
#               num_frames into subfolders of write_dir_path
# folder_paths: list of paths to trial folders containing individual images
# write_dir_path: path to folder to contain subfolders with the chunks 
# patient_number: string containing patient id number
# trial_length: number of individual images in each folder in folder_paths
# num_frames: number of frames to be placed in each chunk
# ext: file extension
# start_idx: optional parameter for fixing things if you mess up
def split_camera(folder_paths, write_dir_path, patient_number, trial_length, num_frames, file_name, ext, start_idx = 0):
    file_paths = list_filenames(folder_paths, trial_length, file_name, ext, start_idx)

    if not os.path.exists(write_dir_path):
        os.mkdir(write_dir_path)
        
    move_files(write_dir_path, num_frames, patient_number, file_paths, file_name, ext, start_idx)

    return

def proc_trial(gt_path, radar_path, thermal_path, trial_num,patient_num, save_path):
    print("Reading GT!")
    gt_dict = pickle.load(open(gt_path, "rb"))
    print("GT Read!")
    gt_dict["OSA"] = (gt_dict["OSA"] > 0.5).astype(int) 
    gt_dict["CSA"] = (gt_dict["CSA"] > 0.5).astype(int) 
    gt_dict["MSA"] = (gt_dict["MSA"] > 0.5).astype(int)
    gt_dict["Hypopnea"] = (gt_dict["Hypopnea"] > 0.5).astype(int)
    dicts_split = split_dicts(gt_dict, list(gt_dict.keys()), 9000)
    print("Reading Radar!")
    radar = np.load(radar_path)
    print("Radar Read!")
    for i in tqdm(range(24)):
        radar_window = radar[i*5*1800:(i+1)*5*1800]
        folder_path = os.path.join(save_path, "v_" + patient_num + "_" + str(120*trial_num + i*5))
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        
        file_path = os.path.join(folder_path, "FMCW_Radar.npy")
        file_path_gt = os.path.join(folder_path, "gt_dict.pkl")
        np.save(file_path, radar_window)

        vid_read_name = "min_" + str(120*trial_num+i*5) + ".tiff"
        vid = read.read_vid(thermal_path, [vid_read_name], shape = (9000,64,64,1), bit_depth="uint16")
        np.save(os.path.join(folder_path, "crop_vid.npy"), vid)
        vid = vid.astype(np.float64)
        signal = np.mean(vid, axis = (1,2))
        signal = (signal - np.mean(signal)) / np.std(signal)
        np.save(os.path.join(folder_path, "signal.npy"), signal)
        plt.plot(signal.flatten())
        plt.savefig(os.path.join(folder_path, "signal.png"))
        plt.close()
        pickle.dump(dicts_split[i], open(file_path_gt, "wb"))