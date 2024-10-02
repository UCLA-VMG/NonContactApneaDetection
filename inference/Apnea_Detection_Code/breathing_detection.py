import numpy as np
import scipy
from scipy import signal
from scipy.fft import fftshift
from scipy.fft import fft, fftfreq

import matplotlib.pyplot as plt
import matplotlib as mpl

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm

import os
import math
import cv2

import numpy as np
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datasets import RFData, CameraData, SignalDataset, FuseDatasets
import utils
import utils_motion as p_lib

def calculate_evaluation_metrics(gt, predicted):
    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(predicted - gt))
    
    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(np.mean((predicted - gt) ** 2))
    
    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((gt - predicted) / gt)) * 100

    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    
    return mae, rmse, mape


def calculate_breathing_rate_dft(signal, sampling_rate, padding_factor=4):
    """
    Calculate the breathing rate from a breathing signal using the Discrete Fourier Transform (DFT) with zero-padding.

    Parameters:
    signal (numpy array): The breathing signal.
    sampling_rate (float): The sampling rate of the signal in Hz.
    padding_factor (int): The factor by which to increase the length of the signal with zero-padding (default is 4).

    Returns:
    float: The breathing rate in breaths per minute.
    """
    # Zero-pad the signal
    padded_length = len(signal) * padding_factor
    padded_signal = np.pad(signal, (0, padded_length - len(signal)), 'constant')
    
    # Perform the DFT of the padded signal
    dft = np.fft.fft(padded_signal)
    
    # Get the frequency bins
    frequencies = np.fft.fftfreq(len(padded_signal), d=1/sampling_rate)
    
    # Compute the power spectrum (magnitude squared of DFT)
    power_spectrum = np.abs(dft) ** 2
    
    # Find the peak in the power spectrum within the valid frequency range (0.1 - 0.45 Hz)
    valid_freq_range = (frequencies > 0.1) & (frequencies < 0.45)
    valid_frequencies = frequencies[valid_freq_range]
    valid_power_spectrum = power_spectrum[valid_freq_range]
    
    # Find the frequency with the highest power
    peak_frequency = valid_frequencies[np.argmax(valid_power_spectrum)]
    
    #calculate uncertainty metric: SNR
    window = 2
    peak_frequency_idx = np.argmax(valid_power_spectrum)+np.where(valid_freq_range)[0][0]
    signal_power = np.sum(power_spectrum[peak_frequency_idx-window:peak_frequency_idx+window+1])
    noise_power = np.sum(power_spectrum[valid_freq_range]) - signal_power
    snr = signal_power / noise_power

    # Calculate the breathing rate in breaths per minute
    breathing_rate = peak_frequency * 60
    
    return breathing_rate, snr

def beamforming_radar(range_bins, AoA_degree):
    tx_id = 0
    AoA = AoA_degree/180*np. pi
    range_bins = range_bins[:,:,:,0] + 1j*range_bins[:,:,:,1] 
    tx_rxs_slice = range_bins[:, tx_id]
    # range_bins.shape, start_idx
    # tx_rxs_slice.shape
    temp_tx_rxs_slice = np.transpose(tx_rxs_slice, axes=(0,2,1))
    temp_tx_rs_slice = temp_tx_rxs_slice * np.exp(2*np.pi*1j*np.sin(AoA)/2*np.arange(0, 4))
    beam_formed_slice = np.transpose(temp_tx_rs_slice, axes=(0,2,1))
    beam_formed_collapsed = np.sum(beam_formed_slice, axis=1)
    return beam_formed_collapsed


def get_unwrapped_wave(radar_data):
    radar_data = radar_data[:,0,0,:,12]
    radar_data = radar_data[:,0] + 1j*radar_data[:,1]
    phase = np.angle(radar_data)
    phase = np.unwrap(phase)
    return phase

def smooth_diff_norm(phase, window_size=25, savgol=True):
# smooth the phase, then differentiate, then normalize
    phase = np.convolve(phase, np.ones(window_size)/window_size, mode='same')
    phase = np.diff(phase)
    phase = (phase - np.mean(phase)) / np.std(phase)
    if(savgol):
        phase = scipy.signal.savgol_filter(phase, 30*3, 3)
    return phase

def apply_to_rangebins(radar_data, func):
    N = radar_data.shape[0]
    new_arr = []
    for i in range(N):
        new_arr.append(func(radar_data[i]))
    return np.array(new_arr)

def calculate_snr_range_bins(radar_data):
    N = radar_data.shape[0]
    snr = np.zeros(N)

    for i in range(N):
        _, snr[i] = calculate_breathing_rate_dft(radar_data[i], 30)
    
    return snr, np.argmax(snr)

def custom_detrend(sig, Lambda=1000):
    """custom_detrend(sig, Lambda) -> filtered_signal
    This function applies a detrending filter.
    This code is based on the following article "An advanced detrending method with application
    to HRV analysis". Tarvainen et al., IEEE Trans on Biomedical Engineering, 2002.
    *Parameters*
      ``sig`` (1d numpy array):
        The sig where you want to remove the trend.
      ``Lambda`` (int):
        The smoothing parameter.
    *Returns*
      ``filtered_signal`` (1d numpy array):
        The detrended sig.
    """
    from scipy import sparse as sp
    signal_length = sig.shape[0]

    # observation matrix
    H = np.identity(signal_length)

    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = sp.spdiags(diags_data, diags_index, (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot((H - np.linalg.inv(H + (Lambda ** 2) * np.dot(D.T, D))), sig)
    return filtered_signal




if __name__ == '__main__':

    np.random.seed(0)
    torch.manual_seed(0)

    data_path = r"data_path"
    trial_folders = os.listdir(data_path)
    train_folders = trial_folders

    thermal_file_name = "Thermal_Camera"
    num_samps_oversample = 20 # per experiment, number of samples to generate
    data_length = 9000
    fs = 30
    out_len = 1800 # sample length generated
    thermal_ext = ".npy"

    dataset_thermal_train = CameraData(data_path, train_folders, thermal_file_name, num_samps_oversample, fs, data_length, out_len, thermal_ext)
    samp_f=5e6
    freq_slope=60.012e12
    samples_per_chirp=256
    num_tx = 3
    num_rx = 4
    radar_file_name = "FMCW_Radar.npy"
    window_size = 25 # number of range bins to use
    dataset_radar_train = RFData(data_path, train_folders, data_length, radar_file_name, out_len, window_size, samples_per_chirp, samp_f, freq_slope, num_samps_oversample, num_tx, num_rx, fs)

    vital_dict_file_name = "gt_dict.pkl"
    vital_key_thermal = "AIR_flow"
    l_freq_bpm = 5
    u_freq_bpm = 30
    dataset_gt_thermal_train = SignalDataset(data_path, train_folders, vital_dict_file_name, vital_key_thermal, data_length, out_len, False, fs, 1024, False, l_freq_bpm, u_freq_bpm, num_samps_oversample, False)
    fused_dataset_train = FuseDatasets([dataset_thermal_train, dataset_radar_train, dataset_gt_thermal_train], ["thermal", "radar", "gt"], out_len=out_len)

    br_thermal = []
    br_gt_airflow = []
    br_radar = []
    br_snr_thermal = []
    br_snr_radar = []
    br_fusion = []

    def smooth_norm(phase, window_size=25, savgol=True):
        phase_diff = np.convolve(phase, np.ones(window_size)/window_size, mode='same')
        phase_diff = np.diff(phase_diff)
        phase_diff = (phase_diff - np.mean(phase_diff)) / np.std(phase_diff)
        if(savgol):
            phase_diff = scipy.signal.savgol_filter(phase_diff, 30*3, 3)
        return phase_diff

    for idx in tqdm(range(len(dataset_thermal_train)), total=len(dataset_thermal_train)):
        data = fused_dataset_train[idx]
        thermal_wave = data['thermal']
        thermal_wave = smooth_diff_norm(thermal_wave)
        
        gt_wave = data['gt'][:,0]
        
        radar_wave = beamforming_radar(data['radar'], 0).transpose()
        radar_wave = np.unwrap(np.angle(radar_wave), -1)
        radar_wave = apply_to_rangebins(radar_wave, smooth_diff_norm)
        snrs, best_idx = calculate_snr_range_bins(radar_wave)
        radar_wave = radar_wave[best_idx]

        br_prediction_radar, br_snr_est_radar = calculate_breathing_rate_dft(radar_wave, 30)
        br_prediction, br_snr_est_thermal = calculate_breathing_rate_dft(thermal_wave, 30)
        br_gt, _ = calculate_breathing_rate_dft(gt_wave, 30)
        #fusion of radar and thermal
        fusion_wave = (radar_wave*br_snr_est_radar + thermal_wave*br_snr_est_thermal*1.5)
        br_fusion.append(calculate_breathing_rate_dft(fusion_wave, 30)[0])

        br_gt_airflow.append(br_gt)
        br_thermal.append(br_prediction)
        br_snr_thermal.append(br_snr_est_thermal)
        br_radar.append(br_prediction_radar)
        br_snr_radar.append(br_snr_est_radar)

    br_thermal = np.array(br_thermal)
    br_radar = np.array(br_radar)
    br_gt_airflow = np.array(br_gt_airflow)
    br_snr_thermal = np.array(br_snr_thermal)
    br_snr_radar = np.array(br_snr_radar)
    br_fusion = np.array(br_fusion)

    calculate_evaluation_metrics(br_gt_airflow, br_thermal)
    calculate_evaluation_metrics(br_gt_airflow, br_radar)
    calculate_evaluation_metrics(br_gt_airflow, br_fusion)
