import numpy as np
import matplotlib.pyplot as plt


def get_motion_scores(signal, dmin=5, dmax=5, dmax_func=None, dmin_func=None, plot=False, time_arr=None, gt_signal=None, include_edges=True):
    """
    Input :
    signal: 1d-array, data signal for which to assign motion scores
    dmin, dmax: int, optional, size of chunks, use this if the signal has a lot of minima/maxima
    plot: bool, optional, true if user wants to plot the relevant graphs
    time_arr: 1d-array, optional, array corresponding time stamps for plotting signal, only used if plot is true
    gt_signal: 1d-array, optional, ground truth signal, only used if plot is true
    include_edges: bool, optional, include the start and end of signal in the motion score calculation if true
    Output :
    lmin,lmax : high/low envelope idx of input signal
    dists_min, dists_max: motion scores associated with each minima/maxima
    """
    lmin, lmax = hl_envelopes_idx(signal=signal, dmin=dmin, dmax=dmax, dmax_func=dmax_func, dmin_func=dmin_func)

    min_env = signal[lmin]
    max_env = signal[lmax]
    half_len = 4 # N-nearest neighbor parameter

    dists_min = []
    dists_max = []

    if((include_edges == True) and (len(max_env) >  half_len)):
        start_vec = [(max_env[0]-max_env[1+i])**2 for i in range(half_len)]
        start_vec = np.array(start_vec)
        dists_max.append(np.sum(start_vec)/((1 or np.mean(start_vec[np.argsort(start_vec)[::-1]]))*len(start_vec)))
        

   
    for i in range(1, len(max_env)-1):
        vec_start = max(0, i - half_len)
        vec_end = min(len(max_env), i + half_len)
        vec = max_env[vec_start:vec_end]
        vec = np.array(vec)
        center_vec = np.repeat(max_env[i], len(vec))
        dist = np.sum((center_vec - vec)**2)/((1 or np.mean(vec[np.argsort(vec)[::-1]]))*len(vec))
        dists_max.append(dist)

    if((include_edges == True) and (len(max_env) >  half_len) and (len(min_env) >  half_len)):
        end_vec = [(max_env[len(max_env)-1]-max_env[len(max_env)-2-i])**2  for i in range(half_len)]
        end_vec = np.array(end_vec)
        dists_max.append(np.sum(end_vec)/((1 or np.mean(end_vec[np.argsort(end_vec)[::-1]]))*len(end_vec)))

        start_vec = [(min_env[0]-min_env[1+i])**2 for i in range(half_len)]
        start_vec= np.array(start_vec)
        dists_min.append(np.sum(start_vec)/((1 or np.mean(start_vec[np.argsort(start_vec)[::-1]]))*len(start_vec)))

    for i in range(1, len(min_env)-1):
        vec_start = max(0, i - half_len)
        vec_end = min(len(min_env), i + half_len)
        vec = min_env[vec_start:vec_end]
        vec = np.array(vec)
        center_vec = np.repeat(min_env[i], len(vec))
        dist = np.sum((center_vec - vec)**2)/((1 or np.mean(vec[np.argsort(vec)[::-1]]))*len(vec))
        dists_min.append(dist)

    if((include_edges == True) and (len(min_env) >  half_len)):
        end_vec = [(min_env[len(min_env)-1]-min_env[len(min_env)-2-i])**2  for i in range(half_len)]
        end_vec = np.array(end_vec)
        dists_min.append(np.sum(end_vec)/((1 or np.mean(end_vec[np.argsort(end_vec)[::-1]]))*len(end_vec)))

    dists_min = np.array(dists_min)
    dists_max = np.array(dists_max)

    if(plot == True):
        if(len(lmin) == 0 or len(lmax) == 0): 
            lx = []
            min_env = []

            mx = []
            max_env = []
        else:
            lx = time_arr[lmin]
            min_env = signal[lmin]

            mx = time_arr[lmax]
            max_env = signal[lmax]
        
        
        plt.plot(lx, min_env, 'ro-', label='low')
        plt.plot(mx, max_env, 'go-', label='high')
        plt.plot(time_arr, signal, color='black', label='Thermal Signal')
        if(gt_signal is not None):
            plt.plot(time_arr, gt_signal, label='gt_signal')
        plt.title("Thermal Signal with Envelope")
        plt.xlabel("Time (s)")
        plt.legend()
        plt.show()
        if((len(dists_min) == len(lx)) and (len(dists_max) == len(mx))):
            plt.plot(lx, dists_min, label='low')
            plt.plot(mx, dists_max, label='high')
            plt.title("Thermal Envelope Point Distance to Neighbors")
            plt.show()
    return(dists_min, dists_max, lmin, lmax)

def find_movement_peaks(dists_min, dists_max, lmin, lmax, time_arr, amplitude_th, sample_idx_threshold, prints=False, plot=False):
    """
    Input :
    dists_min, dists_max: motion scores associated with each minima/maxima
    lmin,lmax : high/low envelope idx of input signal
    time_arr: 1d-array, array corresponding time stamps
    amplitude_th: float, amplitude threshold to seperate motion peaks that need to be removed
    sample_idx_thershold: float, if multiple motion peaks are within a sample index threshold (divide by fs to convert to time) perform suppression by taking the median peak
    plot: bool, optional, true if user wants to plot the relevant graphs
    Output :
    center_list_max: a list of motion peak locations in units of sample index
    """
    time_arr_min = time_arr[lmin]
    idx_min = []
    x_min = []
    y_min = []

    for i in range(len(dists_min)):
        if(dists_min[i] > amplitude_th):
            idx_min.append(lmin[i])
            x_min.append(time_arr_min[i])
            y_min.append(dists_min[i])
    
    center_list_min = []
    cx_min = []
    i = 0
    while((len(x_min) != 0) and (i < len(y_min))):
        offset = 1
        avg_pos_arr = [(x_min[i], y_min[i], idx_min[i])]
        while(((i + offset) < len(y_min)) and (abs(x_min[i] - x_min[i+offset]) < sample_idx_threshold)):
            avg_pos_arr.append((x_min[i+offset], y_min[i+offset], idx_min[i+offset]))
            offset += 1
        
        centers = []
        cx = []
        for xy in avg_pos_arr:
            x_min.remove(xy[0])
            y_min.remove(xy[1])
            idx_min.remove(xy[2])
            centers.append(xy[2])
            cx.append(xy[0])
        center_list_min.append(np.median(centers)) # can change this to be the mean, mode, etc...
        cx_min.append(np.median(cx))

    time_arr_max = time_arr[lmax]
    idx_max = []
    x_max = []
    y_max = []

    for i in range(len(dists_max)):
        if(dists_max[i] > amplitude_th):
            idx_max.append(lmax[i])
            x_max.append(time_arr_max[i])
            y_max.append(dists_max[i])
    
    center_list_max = []
    cx_max = []
    i = 0
    while((len(x_max) != 0) and (i < len(y_max))):
        offset = 1
        avg_pos_arr = [(x_max[i], y_max[i], idx_max[i])]
        while(((i + offset) < len(y_max)) and (abs(x_max[i] - x_max[i+offset]) < sample_idx_threshold)):
            avg_pos_arr.append((x_max[i+offset], y_max[i+offset], idx_max[i+offset]))
            offset += 1
        
        centers = []
        cx = []
        for xy in avg_pos_arr:
            x_max.remove(xy[0])
            y_max.remove(xy[1])
            idx_max.remove(xy[2])
            centers.append(xy[2])
            cx.append(xy[0])
        center_list_max.append(np.median(centers)) # can change this to be the mean, mode, etc...
        cx_max.append(np.median(cx))

    if(prints == True):
        print("Center List Max: ", center_list_max, '\n')
        print("Center List Min: ", center_list_min, '\n')

    for i in range(len(center_list_max)):
        t = time_arr[int(center_list_max[i])]
        for j in range(len(center_list_min)):

            if(prints == True):
                print("t: ", t, "time_arr: ", time_arr[int(center_list_min[j])], '\n')
                print(abs(t - time_arr[int(center_list_min[j])]), sample_idx_threshold, '\n')

            if((len(center_list_min) != 0) and (abs(t - time_arr[int(center_list_min[j])]) < sample_idx_threshold/30)):
                center_list_min.remove(center_list_min[j])
                cx_min.remove(cx_min[j])
                break
    
    center_list_max.extend(center_list_min)
    cx_max.extend(cx_min)

    if(plot == True):
        if(len(center_list_max) != 0):
            plt.scatter(cx_max, center_list_max)
            plt.title("center_list_max")
            plt.show()
        else:
            print("No movement detected")

    return(center_list_max)

def remove_peaks(center_list, sig, half_length=90):
    """
    Input: 
    center_list: list, indices of peaks
    sig: 1d-array, signal
    Output:
    signal_comps: list of 1d-arrays with specified peaks filtered
    """
    signal_comps = []
    last_lower = -1

    if(len(center_list) == 1):
            cut_center = int(center_list[0])
            lower = max(0, cut_center-half_length)
            upper = min(len(sig), cut_center+half_length)
            signal_comps.append([sig[:lower], [0, lower]])
            signal_comps.append([sig[upper:], [upper, len(sig)]])
            return(signal_comps)

    for i in range(len(center_list)):
        cut_center = int(center_list[i])
        lower = max(0, cut_center-half_length)
        upper = min(len(sig), cut_center+half_length)
        
        if(i == 0):
            signal_comps.append([sig[:lower], [0, lower]])
            last_upper = upper
        if((i != (len(center_list) - 1)) and (i != 0)):
            signal_comps.append([sig[last_upper:lower], [last_upper, lower]])
            last_upper = upper
        if(i == (len(center_list) - 1)):
            signal_comps.append([sig[last_upper:lower], [last_upper, lower]])
            signal_comps.append([sig[upper:], [upper, len(sig)]])

    return(signal_comps)

def data_splitter(center_list, dists_min, signal, dists_max, prints=True):
    center_list = sorted(center_list)

    if((len(dists_max) == 0) or (len(dists_min) == 0)):
        max_dist = 350
    else:
        max_dist = int(max([max(dists_min), max(dists_max)])*(90/20))


    half_length = 50
    
    signal_comps = remove_peaks(center_list, signal, half_length=half_length)

    if(prints==True):
        print('max dist: ', max_dist)
        print('center_list', center_list)
        print('signal_comps len',len(signal_comps))

    return(signal_comps)  

def hl_envelopes_idx(signal, dmin=1, dmax=1, dmax_func=None, dmin_func=None, step_size=1, prints=False):
    """
    Input :
    signal: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the signal has a lot of minima/maxima
    dmin_func, dmax_func: function, optional, used to calculate signal dependent chunk size
    step_size: int, optional, stride size for windowing the chunks
    Output :
    lmin,lmax : high/low envelope idx of input signal
    """
    lmin = (np.diff(np.sign(np.diff(signal))) > 0).nonzero()[0] + 1 
    lmax = (np.diff(np.sign(np.diff(signal))) < 0).nonzero()[0] + 1

    dmin_num = dmin
    dmax_num = dmax
    if(dmax_func is not None):
        dmax_num = dmax_func(lmax)

    if(dmin_func is not None):
        dmin_num = dmin_func(lmin)
    
    if(prints == True):
        print('len(lmax): ', len(lmax), '\n')
        print('len(lmin): ', len(lmin), '\n')
        print('step size: ', step_size, '\n')
        print('dmin: ', dmin_num)
        print('dmax: ', dmax_num)
        
    if(dmin_num != 1):
        lmin = lmin[[max(i-dmin_num//2,0)+np.argmin(signal[lmin[max(i-dmin_num//2,0):min(i+dmin_num//2,len(lmin))]]) for i in range(0,len(lmin),step_size)]]
    
    if(dmax_num != 1):
        lmax = lmax[[max(i-dmax_num//2,0)+np.argmax(signal[lmax[max(i-dmax_num//2,0):min(i+dmax_num//2,len(lmax))]]) for i in range(0,len(lmax),step_size)]]
    
    return(lmin,lmax)

def predict(signal, time_arr, th=0.4, dmin=25, dmax=25, dists_min=None, dists_max=None, dmax_func=None, dmin_func=None, mode='mean', percentage=25, plot=False, prints=False):
    """
    Input: 
    signal: 1d-array, data signal
    time_arr: 1d-array, optional, array corresponding time stamps for plotting signal, only used if plot is true
    th: float, threshold value for apnea prediction
    dmin, dmax: int, optional, size of chunks, use this if the signal has a lot of minima/maxima
    dmin_func, dmax_func: function, optional, used to calculate signal dependent chunk size
    mode: str, mode of normalization
    percentage: float, % percentile normalization if mode is not one of 'mean', 'median', or '90th'
    plot: bool, optional, true if user wants to plot the relevant graphs
    prints: bool, optional, true if user wants to print relevant statments
    Output:
    pred: 1d-array, binary prediction
    """
    lmin, lmax = hl_envelopes_idx(signal=signal, dmin=dmin, dmax=dmax, dmax_func=dmax_func, dmin_func=dmin_func, prints=prints)
    
    if(len(lmin) == 0 or len(lmax) == 0):
        return(np.zeros(len(signal)))
        
    max_th = np.interp(time_arr, time_arr[lmax], signal[lmax])
    min_th = np.interp(time_arr, time_arr[lmin], signal[lmin])

    pred = None
    cn = None
    if(mode == 'max'):
        pred = ((max_th - min_th)/max(max_th - min_th) < th).astype(int)
        cn = max(max_th - min_th)
    elif(mode == '90th'):
        pred = ((max_th - min_th)/np.percentile(max_th - min_th, 90) < th).astype(int)
        cn = np.percentile(max_th - min_th, 90)
    elif(mode == 'median'):
        pred = ((max_th - min_th)/np.median(max_th - min_th) < th).astype(int)
        cn = np.median(max_th - min_th)
    elif(mode == 'mean'):
        pred = ((max_th - min_th)/np.mean(max_th - min_th) < th).astype(int)
        cn = np.mean(max_th - min_th)
    else:
        pred = ((max_th - min_th)/np.percentile(max_th - min_th, percentage) < th).astype(int)
        cn = np.percentile(max_th - min_th, percentage)

    if(prints == True):
        print("lmin: ", len(lmin), "lmax: ", len(lmax), '\n')

    if(plot == True):
        
        lx = time_arr[lmin]
        mx = time_arr[lmax]

        min_env = signal[lmin]
        max_env = signal[lmax]

        plt.plot(lx, min_env, 'ro-', label='low env')
        plt.plot(mx, max_env, 'go-', label='high env')
        plt.plot(time_arr, signal, color='black', label='Thermal Video Mean Over Time')
        plt.title("Thermal Signal with Envelope")
        plt.ylabel("Amplitude")
        plt.xlabel("Time (s)")
        plt.legend()
        # plt.axis('off')
        plt.show()

        # if((dists_min is not None) and (dists_max is not None)):
        #     if((len(dists_min) == len(lx)) and (len(dists_max) == len(mx))):
        #         plt.figure(figsize=(50,10))
        #         plt.plot(lx, dists_min, label='low')
        #         plt.plot(mx, dists_max, label='high')
        #         plt.title("Thermal Envelope Point Distance to Neighbors")
        #         plt.show()


        plt.plot(time_arr, (max_th - min_th)/cn, label='Normalized Envelope Difference')
        plt.plot(time_arr, pred, label='Prediction')
        plt.title("Thermal Pred Before Thresholding")
        plt.xlabel("Time (s)")
        plt.ylabel("Normalized Difference")
        plt.legend()
        plt.show()

    return(pred)

def helper_func(thermal, t_arr, gt_breathing, th=0.4, dmin=25, dmax=25, dmax_func=None, dmin_func=None,  mode='mean', plot=False, prints=False, include_edges=True, motion_th=4.5 , is_radar=False):
    dists_min, dists_max, lmin, lmax = get_motion_scores(signal=thermal, dmin=5, dmax=5, dmax_func=dmax_func, dmin_func=dmin_func, plot=False, time_arr=t_arr, gt_signal=gt_breathing, include_edges=include_edges)
    if(is_radar == True):
        center_list = find_movement_peaks(dists_min=dists_min, dists_max=dists_max, lmin=lmin, lmax=lmax, time_arr=t_arr, amplitude_th=motion_th, sample_idx_threshold=150, prints=prints, plot=False)
    else:
        center_list = find_movement_peaks(dists_min=dists_min, dists_max=dists_max, lmin=lmin, lmax=lmax, time_arr=t_arr, amplitude_th=99999, sample_idx_threshold=150, prints=prints, plot=False)
    signal_comps = data_splitter(center_list=center_list, dists_min=dists_min, signal=thermal, dists_max=dists_max, prints=prints)
    pred = np.zeros(len(thermal))

    for sig in signal_comps:
        if(len(sig[0]) != 0):
            std = np.std(sig[0])
            if(std != 0):
                if(prints == True):
                    print('len(sig[0])', len(sig[0]), 'std(sig[0])', np.std(sig[0]))
                
                new_thermal = (sig[0] - np.mean(sig[0]))/std
                pred[sig[1][0]:sig[1][1]] = predict(signal=new_thermal, time_arr=t_arr[sig[1][0]:sig[1][1]], th=th, dmin=dmin, dmax=dmax, dmax_func=dmax_func, dmin_func=dmin_func, dists_min=None, dists_max=None, mode=mode, plot=plot, prints=prints)

    if(len(signal_comps) == 0):
        pred = predict(signal=thermal, time_arr=t_arr, th=th, dmin=dmin, dmax=dmax, dists_min=dists_min, dists_max=dists_max, dmax_func=dmax_func, dmin_func=dmin_func, mode=mode, plot=plot, prints=prints)
    return(pred)

def depth2_apnea_predictor(signal, time_arr, th=0.4, dmin=25, dmax=25, dmax_func=None, dmin_func=None, mode='mean', plot=False, prints=False, gt_signal=None, include_edges=True, motion_th=4.5, is_radar=False):
    dists_min, dists_max, lmin, lmax = get_motion_scores(signal=signal, dmin=5, dmax=5, dmax_func=None, dmin_func=None, plot=plot, time_arr=time_arr, gt_signal=gt_signal, include_edges=include_edges)
    center_list = find_movement_peaks(dists_min=dists_min, dists_max=dists_max, lmin=lmin, lmax=lmax, time_arr=time_arr, amplitude_th=motion_th, sample_idx_threshold=150, prints=prints, plot=False)
    signal_comps = data_splitter(center_list=center_list, dists_min=dists_min, signal=signal, dists_max=dists_max, prints=prints)
    pred = np.zeros(len(signal))
    if(len(signal_comps) == 0):
        pred = predict(signal=signal, time_arr=time_arr, th=th, dmin=dmin, dmax=dmax, dists_min=dists_min, dists_max=dists_max, dmax_func=dmax_func, dmin_func=dmin_func, mode=mode, percentage=25, plot=plot, prints=prints)
    else:        
        for p, sig in enumerate(signal_comps):
            if(len(sig[0]) != 0):
                std = np.std(sig[0])
                if(std != 0):
                    if(prints == True):
                        print(f"{p}th signal component")
                        print('len(sig[0])', len(sig[0]), 'std(sig[0])', np.std(sig[0]))
                        print(r"#################################################")
                    
                    new_signal = (sig[0] - np.mean(sig[0]))/std
                    if(gt_signal is not None):
                        pred_chunk = helper_func(new_signal, time_arr[sig[1][0]:sig[1][1]], gt_signal[sig[1][0]:sig[1][1]], th=th, dmin=dmin, dmax=dmax, dmax_func=dmax_func, dmin_func=dmin_func, mode=mode, plot=plot, prints=prints, motion_th=motion_th, is_radar=is_radar)
                    else:
                        pred_chunk = helper_func(new_signal, time_arr[sig[1][0]:sig[1][1]], None, th=th, dmin=dmin, dmax=dmax, dmax_func=dmax_func, dmin_func=dmin_func, mode=mode, plot=plot, prints=prints, motion_th=motion_th, is_radar=is_radar)

                    if(prints == True):
                        print(r"#################################################")
                    pred[sig[1][0]:sig[1][1]] = pred_chunk
    return(pred)

def apnea_predictor(signal, time_arr, th=0.4, dmin=25, dmax=25, dmax_func=None, dmin_func=None, mode='mean', plot=False, prints=False, gt_signal=None, include_edges=True, motion_th=4.5, split=False):
    if(split == False):
        return(depth2_apnea_predictor(signal, time_arr, th=th, dmin=dmin, dmax=dmax, dmax_func=dmax_func, dmin_func=dmin_func, mode=mode, plot=plot, prints=prints, gt_signal=gt_signal, include_edges=include_edges, motion_th=motion_th))
    else:
        pred = np.zeros(len(signal))
        for i in range(0, 5):
            if(prints == True):
                print(f"{i}th iteration")
            new_signal = signal[i*1800:(i+1)*1800]
            new_time_arr = time_arr[i*1800:(i+1)*1800]
            new_gt_signal = gt_signal[i*1800:(i+1)*1800]
            pred[i*1800:(i+1)*1800] = depth2_apnea_predictor(new_signal, new_time_arr, th=th, dmin=dmin, dmax=dmax, dmax_func=dmax_func, dmin_func=dmin_func, mode=mode, plot=plot, prints=prints, gt_signal=new_gt_signal, include_edges=include_edges, motion_th=motion_th)
        return(pred)

def get_apnea_count(pred, center_th=150, time_th=150, plot=False, lims=None):
    if(plot == True):
        plt.figure(figsize=(50,10))
        plt.plot(np.linspace(0, (len(pred)-1)/30, len(pred)), pred, label='pred before filter')
        if(lims is not None):
            plt.xlim([lims[0],lims[1]])
        plt.legend()
        plt.show()
    i = 0
    center_list = []
    while(i < len(pred)):
        if(pred[i] == 1):
            peak = [i]
            while((i < len(pred)) and (pred[i] == 1)):
                peak.append(i)
                i = i + 1
            center_list.append([np.median(peak), peak[0], peak[-1]])
        else:
            i = i + 1


    k = 1
    while(k < len(center_list)):
        if(abs(center_list[k][1] - center_list[k-1][2]) < center_th):
            pred[center_list[k-1][1]:center_list[k][2]] = 1
        k = k + 1
    
    if(plot == True):
        plt.figure(figsize=(50,10))
        plt.plot(np.linspace(0, (len(pred)-1)/30, len(pred)), pred, label='pred after consolidation')
        if(lims is not None):
            plt.xlim([lims[0],lims[1]])
        plt.legend()
        plt.show()
    
    i = 0
    center_list = []
    while(i < len(pred)):
        if(pred[i] == 1):
            peak = [i]
            while((i < len(pred)) and (pred[i] == 1)):
                peak.append(i)
                i = i + 1
            center_list.append([np.median(peak), peak[0], peak[-1]])
        else:
            i = i + 1

    k = 0
    while(k < len(center_list)):
        if(abs(center_list[k][2] - center_list[k][1]) < time_th):
            pred[center_list[k][1]:center_list[k][2]+1] = 0
        k = k + 1

    if(plot == True):
        plt.figure(figsize=(50,10))
        plt.plot(np.linspace(0, (len(pred)-1)/30, len(pred)), pred, label='pred after suppression')
        if(lims is not None):
            plt.xlim([lims[0],lims[1]])
        plt.legend()
        plt.show()
    
    i = 0
    count = 0
    while(i < len(pred)):
        if(pred[i] == 1):
            while((i < len(pred)) and (pred[i] == 1)):
                i = i + 1
            count = count + 1
        else:
            i = i + 1
    
    if(plot == True):
        plt.figure(figsize=(50,10))
        plt.plot(np.linspace(0, (len(pred)-1)/30, len(pred)), pred, label='pred after filter')
        if(lims is not None):
            plt.xlim([lims[0],lims[1]])
        plt.legend()
        plt.show()
    return(count, pred)

def get_gt_apnea_count(pred): 
    i = 0
    count = 0
    while(i < len(pred)):
        if(pred[i] == 1):
            while((i < len(pred)) and (pred[i] == 1)):
                i = i + 1
            count = count + 1
        else:
            i = i + 1   
    return(count, pred)

    