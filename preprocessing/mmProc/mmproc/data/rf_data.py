from fileinput import filename
# from importlib.metadata import files
from mmproc.data.data import Data
import numpy as np

import mmproc.utils.utils_read as read
import mmproc.utils.utils_write as write
import mmproc.utils.utils_display as display
import mmproc.utils.utils_transform as transform
import mmproc.utils.utils_radar as radar

class RFData(Data):
    def __init__(self, read_dirpath: str, write_dirpath: str, read_files: list, write_files: list, bit_depth: str, modality: str, fps: int, radar_params: tuple = (128, 3, 4, 256), velocity_params: tuple = (5000, 60.012, 77, 100, 60), angle_granularity: int = 128):
        super().__init__(read_dirpath = read_dirpath, write_dirpath = write_dirpath, read_files = read_files, write_files = write_files, bit_depth = bit_depth, modality = modality)
        self.read_dirpath = read_dirpath
        self.write_dirpath = write_dirpath
        self.read_files = read_files
        self.write_files = write_files

        self.bit_depth = bit_depth
        self.modality = modality
        self.fps = fps
        self.radar_params = radar_params
        self.velocity_params = velocity_params
        self.angle_granularity = angle_granularity
        # radar params
        self.num_chirps, self.num_tx, self.num_rx, self.num_adc_samples = self.radar_params
        self.sample_rate, self.freq_slope, self.start_freq, self.idle_time, self.ramp_end_time = self.velocity_params
        # read
        self.data = None
        self.timestamps = None
        self.range_bins = None
        self.range_doppler = None # range_doppler_array
        self.padding = None
        self.theta_bins = None # angle_array
        self.shape = None
        self.status = False
        # range res parameters
        self.range_res = None
        self.range_bin_vals = None
        # angle res parameters
        self.omega = None
        self.angle_bin_vals = None
        # velocity res parameters
        self.velocity_res = None
        self.velocities = None

        # other
        self.kargs = {'fps': self.fps}
        self.read()
        print("shape: ", self.shape)   
    
    def __del__(self) -> None:
        self.release_data()
        # print("Released {} resources.".format(RFData))

    ######### reading (creating), writing (saving), displaying and releasing data
    
    def read(self) -> bool:
        self.data, _ = read.read_rf_data(self.read_dirpath, self.read_files, self.radar_params)
        self.shape = self.data.shape
        return True

    def write(self) -> bool:
        print("Writing!")
        # self.status = write.write_txt(self.write_dirpath, self.write_files, self.timestamps)
        self.status = write.write_rf_data(self.write_dirpath, self.write_files, self.data)
        self.check_status()
        return True

    def display(self, display_type, frame_idx, convert) -> bool:
        self.get_range_resolution()
        self.get_velocity_resolution()
        self.get_angle_resolution()
        self.status = display.display_rf_data(display_type, self.velocities, self.angle_bin_vals, self.range_bin_vals, self.theta_bins, self.range_doppler, frame_idx, convert)
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

    ######### display helper functions 



    def get_range_resolution(self, bandwidth=3600.72):
        bandwidth *= 1e6
        self.range_res = 3e8 / (2 * bandwidth)
        self.range_bin_vals = np.arange(self.data.shape[3])*self.range_res
    
    def get_velocity_resolution(self):
        self.velocity_res = 3e8 / (2 * self.start_freq * 1e9 * (self.idle_time + self.ramp_end_time) * 1e-6 * self.num_chirps * self.num_tx)
        self.velocities = np.arange(self.num_chirps) - (self.num_chirps // 2)
        self.velocities = self.velocities * self.velocity_res

    def get_angle_resolution(self):
        self.range_bins = np.fft.fft(self.data, axis=3)
        self.range_doppler = np.fft.fft(self.range_bins, axis=1)
        self.padding = ((0,0), (0,0), (0, self.angle_granularity - self.range_bins.shape[2]), (0,0))
        self.theta_bins = np.fft.fft( np.pad(self.range_doppler, self.padding), axis=2)
        self.omega = np.fft.fftfreq(self.theta_bins.shape[2])*2
        self.angle_bin_vals = np.arcsin(self.omega)
