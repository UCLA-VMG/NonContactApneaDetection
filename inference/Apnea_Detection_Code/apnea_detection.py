import scipy
from scipy import signal
from scipy.fft import fftshift
from scipy.fft import fft, fftfreq

import matplotlib as mpl

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

# smooth the phase, then differentiate, then normalize
def smooth_diff_norm(phase, window_size=25, savgol=True):
    phase_smoothed = np.convolve(phase, np.ones(window_size)/window_size, mode='same')
    phase_diff = np.diff(phase_smoothed)
    phase_diff = (phase_diff - np.mean(phase_diff)) / np.std(phase_diff)
    # if(savgol):
    #     phase_diff = scipy.signal.savgol_filter(phase_diff, 30*3, 3)
    return phase_diff

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

    # second-order difference matrix

    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = sp.spdiags(diags_data, diags_index, (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot((H - np.linalg.inv(H + (Lambda ** 2) * np.dot(D.T, D))), sig)
    return filtered_signal

def linear_func(m, b, n, length_mode=False):
    if(length_mode == True):
        def func(x):
            return(max(int(m*(x**n) + b), 1))
        return(func)
    else:
        def func(x):
            return(max(int(m*(len(x)**n) + b), 1))
        return(func)
    
def get_stats(pred, gt):
    tp = np.sum(pred * gt)
    fp = np.sum(pred * (1-gt))
    tn = np.sum((1-pred) * (1-gt))
    fn = np.sum((1-pred) * gt)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn)/ (tp + tn + fp + fn)
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    confusion_matrix = np.array([[tp, fn], [fp, tn]])
    return precision, recall, accuracy, confusion_matrix

def intraclass_correlation(x, y):
    assert(len(x) != 0)
    mu_x = 0
    for i in range(len(x)):
        mu_x = mu_x + (x[i] + y[i])
    mu_x = mu_x/(2*len(x))

    s_sq = 0
    for i in range(len(x)):
        s_sq  = s_sq  + (x[i] - mu_x)**2 +  (y[i] - mu_x)**2
    s_sq = s_sq/(2*len(x))

    r = 0
    
    if(s_sq == 0):
        return(0)
    for i in range(len(x)):
        r = r + (x[i] - mu_x)*(y[i] - mu_x)
    r = r/(len(x)*s_sq)
    return(r)


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    data_path = "data_path"


    # Define the dataset
    trial_folders = os.listdir(data_path)
    train_folders = trial_folders

    thermal_file_name = "Thermal_Camera"
    num_samps_oversample = 20 # None # per experiment, number of samples to generate
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
    # vital_key_thermal = "CHEST"
    l_freq_bpm = 5
    u_freq_bpm = 30
    dataset_OSA_train = SignalDataset(data_path, train_folders, vital_dict_file_name, 'OSA', data_length, out_len, False, fs, 1024, False, l_freq_bpm, u_freq_bpm, num_samps_oversample, False,
                                normalize=False)
    dataset_CSA_train = SignalDataset(data_path, train_folders, vital_dict_file_name, 'CSA', data_length, out_len, False, fs, 1024, False, l_freq_bpm, u_freq_bpm, num_samps_oversample, False,
                                normalize=False)
    dataset_MSA_train = SignalDataset(data_path, train_folders, vital_dict_file_name, 'MSA', data_length, out_len, False, fs, 1024, False, l_freq_bpm, u_freq_bpm, num_samps_oversample, False,
                                normalize=False)
    dataset_gt_thermal_train = SignalDataset(data_path, train_folders, vital_dict_file_name, vital_key_thermal, data_length, out_len, False, fs, 1024, False, l_freq_bpm, u_freq_bpm, num_samps_oversample, False)
    dataset_gt_radar_train = SignalDataset(data_path, train_folders, vital_dict_file_name, "CHEST", data_length, out_len, False, fs, 1024, False, l_freq_bpm, u_freq_bpm, num_samps_oversample, False)
    fused_dataset_train = FuseDatasets([dataset_thermal_train, dataset_radar_train, dataset_gt_thermal_train, dataset_gt_radar_train, dataset_OSA_train, dataset_CSA_train, dataset_MSA_train], ["thermal", "radar", "gt", "gt_radar", "OSA", "CSA", "MSA"], out_len=out_len)

    window_max = 23
    window_min = 23
    mode = 'mean'

    dmin_func = linear_func(2, 3, 0.45, length_mode=False)
    plot_count = 0 
    plot_max = 20

    gt_apnea_arr = []
    pred_thermal_arr = []
    pred_radar_arr = []


    motion_detection = False

    for idx in tqdm(range(len(dataset_thermal_train)), total=len(dataset_thermal_train)):
        data = fused_dataset_train[idx]
        thermal_wave = data['thermal']
        thermal_wave = smooth_diff_norm(thermal_wave)
        data = fused_dataset_train[idx]
        gt_wave = data['gt'][:,0]
        radar_wave = beamforming_radar(data['radar'], 0).transpose()
        radar_wave1 = np.unwrap(np.angle(radar_wave), -1)
        radar_wave = apply_to_rangebins(radar_wave1, smooth_diff_norm)
        snrs, best_idx = calculate_snr_range_bins(radar_wave)
        radar_wave = radar_wave[best_idx]


        type_osa = None
        gt_OSA = data['OSA']
        gt_CSA = data['CSA']
        gt_MSA = data['MSA']
        gt_apnea = np.logical_or(gt_OSA, gt_CSA, gt_MSA)
        if(np.mean(gt_MSA) > 0):
            type_osa = "MSA"
        if(np.mean(gt_OSA) > 0):
            type_osa = "OSA"
        if(np.mean(gt_CSA) > 0):
            type_osa = "CSA"

        thermal = utils.highpass_filter(thermal_wave, fs=30, fc=0.1/2, order=6)
        thermal = (thermal - np.mean(thermal))/np.std(thermal)
        dists_min, dists_max, lmin, lmax = utils.movement_detector_thermal(thermal, dmin=23, dmax=23, gt_breathing=thermal, plot=False)
        movement = False
        if((max(dists_min) > 2.5) or (max(dists_max) > 2.5)):
            movement = True

        if((movement == False)):
            pred_thermal = utils.predict(thermal, dmin=window_min, dmax=window_max, dmin_func=dmin_func, dmax_func=dmin_func, th=0.35, mode=mode, plot=False)

            radar_wave = utils.highpass_filter(radar_wave, fs=30, fc=0.1/2, order=6)
            radar_wave = (radar_wave - np.mean(radar_wave))/np.std(radar_wave)
            pred_radar = utils.predict(radar_wave, dmin=window_min, dmax=window_max, dmin_func=None, dmax_func=None, th=0.4, mode=mode, plot=False)
            _ , pred_radar = p_lib.get_apnea_count(pred_radar, center_th=0, time_th=50, plot=False, lims=None)

            if(np.mean(gt_apnea)> 0.1):
                gt_apnea_arr.append(1)
            else:
                gt_apnea_arr.append(0)
            if(np.mean(pred_thermal) > 0.1):
                pred_thermal_arr.append(1)
            else:
                pred_thermal_arr.append(0)
            if(np.mean(pred_radar) > 0.1):
                pred_radar_arr.append(1)
            else:
                pred_radar_arr.append(0)

        elif((movement == True) and (motion_detection == True)):
            pred_thermal = p_lib.depth2_apnea_predictor(signal=thermal, time_arr=t_arr, th=0.41, dmin=window_min, dmax=window_max, mode=mode, plot=False, prints=False, gt_signal=None, include_edges=True, motion_th=20, is_radar=True) # is_radar = True for a more restrictive motion detection
            _ , pred_radar = p_lib.get_apnea_count(pred_radar, center_th=0, time_th=180, plot=False, lims=None)

            radar_wave = utils.highpass_filter(radar_wave, fs=30, fc=0.1/2, order=6)
            radar_wave = (radar_wave - np.mean(radar_wave))/np.std(radar_wave)

            pred_radar = p_lib.depth2_apnea_predictor(signal=radar_wave, time_arr=t_arr, th=0.49, dmin=window_min, dmax=window_max, mode=mode, plot=False, prints=False, gt_signal=None, include_edges=True, motion_th=20, is_radar=True)
            _ , pred_radar = p_lib.get_apnea_count(pred_radar, center_th=0, time_th=190, plot=False, lims=None)


            if(np.mean(gt_apnea)> 0.1):
                gt_apnea_arr.append(1)
            else:
                gt_apnea_arr.append(0)
            if(np.mean(pred_thermal) > 0.1):
                pred_thermal_arr.append(1)
            else:
                pred_thermal_arr.append(0)
            if(np.mean(pred_radar) > 0.1):
                pred_radar_arr.append(1)
            else:
                pred_radar_arr.append(0)


    print(r"precision, recall, accuracy, confusion_matrix")
    if(motion_detection == False):
        print("Thermal (with no motion): ", get_stats(np.array(pred_thermal_arr), np.array(gt_apnea_arr)))
        print("Radar (with no motion): ", get_stats(np.array(pred_radar_arr), np.array(gt_apnea_arr)))
    else:
        print("Thermal (with motion detection): ", get_stats(np.array(pred_thermal_arr), np.array(gt_apnea_arr)))
        print("Radar (with motion detection): ", get_stats(np.array(pred_radar_arr), np.array(gt_apnea_arr)))
    

    if(motion_detection == False):
        print("Thermal (with no motion)")
    else:
        print("Thermal (with motion detection)")

    precision, recall, accuracy, confusion_matrix = get_stats(np.array(pred_thermal_arr), np.array(gt_apnea_arr))
    print("gt_apneas: ", np.sum(gt_apnea_arr), "pred_apneas: ", np.sum(pred_thermal_arr), "total: ", len(gt_apnea_arr))
    print('interclass_correlation: ', intraclass_correlation(pred_thermal_arr, gt_apnea_arr))
    precision, recall, accuracy, confusion_matrix = utils.get_stats(np.array(pred_thermal_arr), np.array(gt_apnea_arr))                    
    print('F1-score: ' , 2*precision*recall / (precision + recall))

    if(motion_detection == False):
        print("Radar (with no motion)")
    else:
        print("Radar (with motion detection)")
        
    precision, recall, accuracy, confusion_matrix = get_stats(np.array(pred_radar_arr), np.array(gt_apnea_arr))
    print("gt_apneas: ", np.sum(gt_apnea_arr), "pred_apneas: ", np.sum(pred_radar_arr), "total: ", len(gt_apnea_arr))
    print('interclass_correlation: ', intraclass_correlation(pred_radar_arr, gt_apnea_arr))
    precision, recall, accuracy, confusion_matrix = utils.get_stats(np.array(pred_radar_arr), np.array(gt_apnea_arr))                    
    print('F1-score: ' , 2*precision*recall / (precision + recall))