import numpy as np
import pandas as pd
import lxml

def add_apnea_dicts(html_path, signals_raw):
    read_P1 = pd.read_html(html_path)
    ap = [] # holds table id and row id for start of apnea events
    pap = [] # holds table id and row id for start of partial apnea events
    second_apnea_type = ""
    for table_id in  range (2, len(read_P1), 1):# first two tables are patient personal information tables
        sleep_apnea = read_P1[table_id].drop(labels = [3, 4, 5, 6], axis = 1) # we only want epoch (0), start(1), duration(2) columns
        sleep_apnea_array = np.asarray(sleep_apnea)

        for row_id in  range(sleep_apnea_array.shape[0]):
            if(sleep_apnea_array[row_id][0] in ["Apnea Events"]):
                ap.append((table_id,row_id))
            
            if sleep_apnea_array[row_id][0] in ["Partial Apnea Events", "Hypopnea Events"]:
                second_apnea_type = sleep_apnea_array[row_id][0]
                pap.append((table_id,row_id))
                

    # print(ap)
    # print(pap)
    ap_epoch_list = []
    ap_start_list = []
    ap_duration_list = []
    ap_type_list = []
    for table_id in range(2, pap[0][0] + 1,1):
        table_array = np.asarray(read_P1[table_id].drop(labels = [4, 5, 6], axis = 1))
        if table_id == pap[0][0]:
            for char in (table_array[ap[0][1] + 1:pap[0][1],3]):
                ap_type_list.append(char)
            ap_epoch_list.append(table_array[ap[0][1] + 1:pap[0][1],0])
            ap_start_list.append(table_array[ap[0][1] + 1:pap[0][1],1])

            ap_duration_list.append(table_array[ap[0][1] + 1:pap[0][1],2])
        if table_id < pap[0][0]:
            for char in table_array[ap[0][1] + 1:,3]:
                ap_type_list.append(char)
            ap_epoch_list.append(table_array[ap[0][1] + 1:,0])
            ap_start_list.append(table_array[ap[0][1] + 1:,1])
            ap_duration_list.append(table_array[ap[0][1] + 1:,2])

    #print(ap_type_list)

    papnea_epoch_list = []
    papnea_start_list = []
    papnea_duration_list = []
    for table_id in range(2, len(read_P1),1):
        table_array = np.asarray(read_P1[table_id].drop(labels = [3, 4, 5, 6], axis = 1))
        if table_id == pap[0][0]:
            papnea_epoch_list.append(table_array[pap[0][1] + 1:,0])
            papnea_start_list.append(table_array[pap[0][1] + 1 : ,1])
            papnea_duration_list.append(table_array[pap[0][1] + 1:,2])
        if table_id > pap[0][0]:
            papnea_epoch_list.append(table_array[1:,0])
            papnea_start_list.append(table_array[1:,1])
            papnea_duration_list.append(table_array[1:,2])

    ap_ep_ar = np.concatenate(ap_epoch_list,axis = 0).astype(int)
    ap_start_ar = np.concatenate(ap_start_list, axis = 0)
    ap_dur_ar = np.concatenate(ap_duration_list, axis = 0)
    pap_ep_ar = np.concatenate(papnea_epoch_list,axis = 0).astype(int)
    pap_start_ar = np.concatenate(papnea_start_list, axis = 0)
    pap_dur_ar = np.concatenate(papnea_duration_list, axis = 0)
    ap_start_ar = np.array([i[:-1] for i in ap_start_ar]).astype(float)
    ap_dur_ar = np.array([i[:-1] for i in ap_dur_ar]).astype(float)
    pap_start_ar = np.array([i[:-1] for i in pap_start_ar]).astype(float)
    pap_dur_ar = np.array([i[:-1] for i in pap_dur_ar]).astype(float)

    osa = np.zeros((len(signals_raw["ABD"][1])))
    csa = np.zeros((len(signals_raw["ABD"][1])))
    msa = np.zeros((len(signals_raw["ABD"][1])))
    pa = np.zeros((len(signals_raw["ABD"][1])))
    hyp = np.zeros((len(signals_raw["ABD"][1])))

    for i in range(ap_ep_ar.shape[0]):
        begin = int((ap_ep_ar[i]-1) * 6000 + ap_start_ar[i] *200)
        end = int(begin + ap_dur_ar[i] * 200)
        if ap_type_list[i] == "O":
            osa[begin : end] = 1
        elif ap_type_list[i] == "C":
            csa[begin : end] = 1
        else:
            msa[begin : end] = 1

    for i in range(pap_ep_ar.shape[0]):
        sec_begin = int((pap_ep_ar[i]-1) * 6000 + pap_start_ar[i] * 200)
        sec_end = int(sec_begin + pap_dur_ar[i] * 200)
        if second_apnea_type in ["Partial Apnea Events"]:
            pa[sec_begin : sec_end] = 1
        else:
            hyp[sec_begin : sec_end] = 1
    
    all_vitals_new = {}
    for keys in signals_raw.keys():
        all_vitals_new[keys] = signals_raw[keys]
    all_vitals_new["OSA"] = (osa, signals_raw["ABD"][1])
    all_vitals_new["CSA"] = (csa, signals_raw["ABD"][1])
    all_vitals_new["MSA"] = (msa, signals_raw["ABD"][1])
    all_vitals_new["Partial_Apnea"] = (pa, signals_raw["ABD"][1])
    all_vitals_new["Hypopnea"] = (hyp, signals_raw["ABD"][1])

    return all_vitals_new
