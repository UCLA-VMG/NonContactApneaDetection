import os
import pickle
import numpy as np 
import imageio
import scipy.signal as sig
from torch.utils.data import Dataset
from tqdm import tqdm
import rf.organizer as org
from rf.proc import create_fast_slow_matrix, find_range

class CameraData(Dataset):
    def __init__(self, root_path, trial_folders, data_file_name="NIR_Camera", \
                 num_samps_oversample=30, fs=30, data_length = 900, out_len = 64, \
                 extension='.png', get_vid=False) -> None:
        # Essential attributes.
        self.root_path = root_path
        self.trial_folders = trial_folders
        # Number of samples to be created by oversampling one trial.
        self.num_samps_oversample = num_samps_oversample
        # Name of the files being read. Name depends on how the file was save. We have saved the file as rgbd_rgb.
        self.data_file_name = data_file_name
        # Number of frames in the input video. (Requires all data-samples to have the same number of frames).
        self.data_length = data_length
        # Number of frames in the output tensor sample.
        self.out_len = out_len
        # Sampling rate of the camera.
        self.fs = fs
        # File extension format.
        self.ext = extension
        
        self.get_vid = get_vid

        # Remove the folders.
        remove_folders = [fol for fol in self.trial_folders if not os.path.exists(os.path.join(self.root_path, fol))]
        for i in remove_folders:
            self.trial_folders.remove(i)    
            print("Removed", i)

        # Create a list of video number and valid frame number to extract the data from.
        self.video_nums = np.arange(0, len(self.trial_folders))
        # Create all possible sampling combinations.
        self.oversampling_idxs = []
        for num in self.video_nums:
            # Generate the start index.
            if(self.num_samps_oversample is None):
                cur_frame_nums = [0]
            else:
                cur_frame_nums = np.random.randint(low=0, high=self.data_length-self.out_len, size=self.num_samps_oversample)
            # Append all the start indices.
            for cur_frame_num in cur_frame_nums:
                self.oversampling_idxs.append((num,cur_frame_num))
        self.oversampling_idxs = np.array(self.oversampling_idxs, dtype=int)
            
            
    def __len__(self):
        return int(len(self.oversampling_idxs))
        
    def __getitem__(self, idx):
        # Get the video number and the starting frame index.
        video_number, frame_start = self.oversampling_idxs[idx]
        # Get video frames from pngs for the output video tensor.
        if(self.get_vid == True):
            image_path = os.path.join(self.root_path, str(self.trial_folders[video_number]), r"crop_vid.npy")
            thermal_vid = np.load(image_path)[frame_start:(frame_start+self.out_len),:,:,0]
            return(thermal_vid)

        
        if(self.ext == r".tiff"):
            item = []
            for img_idx in range(self.out_len):
                print('image_path ', self.root_path, str(self.trial_folders[video_number]), f"{self.data_file_name}_{frame_start+img_idx}{self.ext}")
                
                image_path = os.path.join(self.root_path, str(self.trial_folders[video_number]), f"{self.data_file_name}_{frame_start+img_idx}{self.ext}")
                item.append(imageio.imread(image_path))
            
            item = np.array(item)
            # Add channel dim if no channels in image.
            if(len(item.shape) < 4): 
                item = np.expand_dims(item, axis=3)
            item = np.transpose(item, axes=(3,0,1,2))
            # Patch for the torch constructor. uint16 is a not an acceptable data-type.
            if(item.dtype == np.uint16):
                item = item.astype(np.int32)
            
            return(np.array(item))

        if(self.ext == r".npy"):
            image_path = os.path.join(self.root_path, str(self.trial_folders[video_number]), r"signal.npy")
            thermal_signal = np.load(image_path)[frame_start:(frame_start+self.out_len),0]
            return(thermal_signal)
        return(None)
        


class RFData(Dataset):
    def __init__(self, root_path, trial_folders, \
                data_length=900, data_file_name="FMCW_Radar", \
                out_len=512, window_size=5, samples=256, \
                samp_f=5e6, freq_slope=60.012e12, num_samps_oversample = 30, \
                num_tx=1, num_rx=1, fs=30) -> None:
        # Root path with the dataset.
        self.root_path = root_path
        self.trial_folders = trial_folders
        self.data_file_name = data_file_name
        
        # Number of samples to be created by oversampling one trial.
        self.num_samps_oversample = num_samps_oversample

        # Remove the folders.
        remove_folders = [fol for fol in self.trial_folders if not os.path.exists(os.path.join(self.root_path, fol))]
        for i in remove_folders:
            self.trial_folders.remove(i)    
            print("Removed", i)

        # The ratio of the sampling frequency of the RF signal and the PPG signal.
        self.fs = fs
        
        # Save the RF config parameters.
        self.window_size = window_size
        self.samples = samples
        self.samp_f = samp_f
        self.freq_slope = freq_slope

        # Window the PPG and the RF samples.
        self.data_length = data_length
        self.out_len = out_len
        self.rf_file_nums = np.arange(len(self.trial_folders))

        self.oversampling_idxs = []
        for num in self.rf_file_nums:
            if(self.num_samps_oversample is None):
                rf_cur_frame_nums = [0]
            else:
                rf_cur_frame_nums = np.random.randint(
                low=0, high = self.data_length - out_len, size = self.num_samps_oversample)
            
            for rf_frame_num in rf_cur_frame_nums:
                self.oversampling_idxs.append((num,(rf_frame_num)))
        self.oversampling_idxs = np.array(self.oversampling_idxs, dtype=int)

        # High-ram, compute FFTs before starting training.
        self.rf_data_list = []
        for rf_file in tqdm(self.trial_folders):
            # Read the raw RF data (frames, num_chirps * num_tx, num_rx, adc_samples)
            frames = np.load(os.path.join(self.root_path, rf_file, self.data_file_name), allow_pickle=True)

            # Process the organized RF data.
            data_f = create_fast_slow_matrix(frames, num_tx, num_rx)
            range_index = find_range(data_f, self.samp_f, self.freq_slope, self.samples)
            # Get the windowed raw data for the network.
            while((range_index-self.window_size//2) < 0):
                range_index += 1
            assert((range_index+self.window_size//2+1) <= data_f.shape[-1])
            raw_data = data_f[..., range_index-self.window_size//2:range_index+self.window_size//2 + 1]
            # Note that item is a complex number due to the nature of the algorithm used to extract and process the pickle file.
            # Hence for simplicity we separate the real and imaginary parts into 2 separate channels.
            raw_data = np.array([np.real(raw_data),  np.imag(raw_data)]) # (2, L, Tx, Rx, window)
            raw_data = np.transpose(raw_data, axes=(1,2,3,0,4))
            self.rf_data_list.append(raw_data)

    def __len__(self):
        return int(len(self.oversampling_idxs))

    def __getitem__(self, idx):
        # Currently only TX and RX = 1.
        file_num, rf_start = self.oversampling_idxs[idx]
        # Get the RF data.
        data_f = self.rf_data_list[file_num]
        data_f = data_f[rf_start : rf_start + self.out_len]
        return data_f

class SignalDataset(Dataset):
    def __init__(self, root_path, trial_folders, vital_dict_file, 
                 vital_key="PPG", data_length=900, out_len=64, \
                 compute_fft=False, fs=30, nfft=1024, \
                 limit_freq=False, l_freq_bpm=45, u_freq_bpm=180, \
                 num_samps_oversample=None, detrend=False, normalize=True) -> None:
        # Root path with the dataset.
        self.root_path = root_path
        # List with the folder names.
        self.trial_folders = trial_folders
        # Filename for the vital signal.
        self.vital_dict_file = vital_dict_file
        # Key for the vital dictionary.
        self.vital_key = vital_key
        # Sampling rate of the vital signal.
        self.fs = fs

        # FFT Attributes 
        self.compute_fft = compute_fft
        self.nfft = nfft
        self.limit_freq = limit_freq
        self.l_freq_bpm = l_freq_bpm
        self.u_freq_bpm = u_freq_bpm
        
        # Length of the vital signal to be taken.
        self.data_length = data_length
        # Output signal length.
        self.out_len = out_len
        # Number of samples to be created by oversampling one trial.
        self.num_samps_oversample = num_samps_oversample
        # Flag to detrend the vital signal.
        self.detrend = detrend

        self.signal_list = []
        # Load signals.
        remove_folders = []
        for folder in self.trial_folders:
            folder_path = os.path.join(self.root_path, folder)
            # Make a list of the folder that do not have the PPG signal.
            if(os.path.exists(os.path.join(folder_path, f"{self.vital_dict_file}"))):
                signal = pickle.load(open(os.path.join(folder_path, f"{self.vital_dict_file}"), 'rb'))[self.vital_key]
                if self.detrend:
                    signal = sig.detrend(signal)
                self.signal_list.append(signal[:self.data_length])
            else:
                remove_folders.append(folder)
        # Remove the PPGs.
        for i in remove_folders:
            self.trial_folders.remove(i)    
            print("Removed", i)

        # Extract the stats for the vital signs.
        self.signal_list = np.array(self.signal_list)
        self.vital_mean = np.mean(self.signal_list, axis=1, keepdims=True)
        self.vital_std = np.std(self.signal_list, axis=1, keepdims=True)
        if(normalize):
            self.signal_list = (self.signal_list - self.vital_mean)/(self.vital_std + 10e-7)
        
        # Create all possible sampling combinations.
        self.oversampling_idxs = []
        if self.num_samps_oversample is not None:
            for num in range(len((self.trial_folders))):
                # Generate the start index.
                cur_frame_nums = np.random.randint(low=0, high=self.data_length-self.out_len, size=self.num_samps_oversample)
                # Append all the start indices.
                for cur_frame_num in cur_frame_nums:
                    self.oversampling_idxs.append((num,cur_frame_num))
        else:
            for num in range(len((self.trial_folders))):
                self.oversampling_idxs.append((num,0))
        self.oversampling_idxs = np.array(self.oversampling_idxs, dtype=int)
            
    def __len__(self):
        return len(self.oversampling_idxs)

    def __getitem__(self, idx):
        # Get signal.
        file_number, frame_start = self.oversampling_idxs[idx]
        # print(file_number, frame_start, frame_start+self.out_len)
        # print(self.trial_folders[file_number])
        item_sig = self.signal_list[int(file_number)][int(frame_start):int(frame_start+self.out_len)]
        return np.array(item_sig, dtype=np.float32)[...,np.newaxis]

    def lowPassFilter(self, BVP, butter_order=4):
        [b, a] = sig.butter(butter_order, [self.l_freq_bpm/60, self.u_freq_bpm/60], btype='bandpass', fs = self.fs)
        filtered_BVP = sig.filtfilt(b, a, np.double(BVP))
        return filtered_BVP
    
class FuseDatasets(Dataset):
    def __init__(self, datasets=[], dataset_keys=[], chosen_dataset_idx=0, out_len=900) -> None:
        # Make sure that the datasets and the keys passed are the same length.
        assert len(dataset_keys) == len(datasets), "Number of Dataset objects != Number of Dataset keys"
        # List of datasets objects.
        self.datasets = datasets
        # List of datasets keys.
        self.dataset_keys = dataset_keys
        # The main dataset to copy the oversampling_idxs.
        self.chosen_dataset_idx = chosen_dataset_idx
        # Make sure all the datasets are compatible before fusing.
        self.assert_compatibility()
        # Merge/Fuse the datasets, i.e. copy oversampling_idxs across all datasets.
        self.merge()
        # save output length
        self.out_len = out_len
            
    def __len__(self):
        return len(self.datasets[0].oversampling_idxs)

    def __getitem__(self, idx):
        item = {}

        for key, dset in zip(self.dataset_keys, self.datasets):
            item[key] = dset[idx]
            if(key == "radar"):
                # if(dset[idx].shape[0] != 2**10):
                if(dset[idx].shape[0] != self.out_len):
                    #sample random idx 
                    idx = np.random.randint(0, len(dset))
                    return self.__getitem__(idx)
        return item
    
    def assert_compatibility(self):
        trial_folders = self.datasets[0].trial_folders
        data_length = self.datasets[0].data_length
        fs = self.datasets[0].fs
        # Make sure the duration and the folder list are the same across all datasets.
        for dset in self.datasets:
            assert data_length * fs == dset.data_length * dset.fs, "The inputs are of different duration. \
                                                                    duration = number of samples * sampling rate."
            assert trial_folders ==  dset.trial_folders, "List of folders are not the same."
    
    def merge(self):
        chosen_dataset = self.datasets[self.chosen_dataset_idx]
        chosen_fs = chosen_dataset.fs
        chosen_oversampling_idxs = chosen_dataset.oversampling_idxs.copy()
        # Copy the oversampling_idxs from the chosen dataset to all other datasets.
        for dset in self.datasets:
            # Account for different sampling rates
            target_oversampling_idxs = []
            for target_file_num, target_frame_start in chosen_oversampling_idxs:
                target_frame_start = int(target_frame_start * (dset.fs / chosen_fs))
                target_oversampling_idxs.append((target_file_num, target_frame_start))
            dset.oversampling_idxs = np.array(target_oversampling_idxs)
