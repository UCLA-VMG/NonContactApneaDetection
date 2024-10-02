import os
import shutil
import warnings
import tqdm
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt

import os
import shutil
import json
import copy
import pickle
from typing import Callable, List
from collections import MutableMapping
from contextlib import suppress
import warnings

import my_utils.utils_read as read
import my_utils.utils_write as write

############################################### dict and Metadata helper functions 

def check_keys_in_dicts(dicts: tuple, key_list: str):
    counter = 0
    for dict in dicts:
        for key in key_list:
            if key not in dict:
                counter += 1
    if counter == len(key_list):
        raise Exception(f"Sorry, no key in key_list {key_list} exists in dictionary {dict}")

def get_meta_data_paths(read_dirpath: str, write_dirpath: str, files: list) -> str:
    if (r"meta_data.json" in files):
        src_file_path = os.path.join(read_dirpath, r"meta_data.json")
        file_path = os.path.join(write_dirpath, r"meta_data.json")
    else:
        raise Exception(f"No meta_data.josn file found in {read_dirpath}!")
    return src_file_path, file_path
            
def del_single_key_value_pair(dictionary: dict, keys: list):
    dict_foo = dictionary
    for key in keys:
        if (key in dict_foo.keys() ):
            value = dict_foo[key]
            if len(keys) != 1:
                keys.remove(key)
                del_single_key_value_pair(value, keys)
            else:
                with suppress(KeyError):
                    del dictionary[key]
        else:
            raise Exception(f"key {key} not in dictionary {dictionary}! Please check your keys list!")
    return dictionary

def del_multiple_key_value_pairs(dictionary: dict, keys: list):
    dict_foo = dictionary.copy()
    for key in keys:
        for value in dict_foo.values():
            if isinstance(value, MutableMapping):
                del_multiple_key_value_pairs(value, keys)
            else:
                with suppress(KeyError):
                    del dictionary[key]
    return dictionary

def remove_single_key_value_pair(dictionary: dict, keys: list):
    keys_copy = copy.deepcopy(keys)
    dictionary = del_single_key_value_pair(dictionary, keys)
    return dictionary

def remove_multiple_key_value_pairs(dictionary: dict, keys: list):
    keys_copy = copy.deepcopy(keys)
    dictionary = del_multiple_key_value_pairs(dictionary, keys)
    return dictionary
            
def append_single_key_value_pair(dictionary: dict, keys: list, value):
    dict_foo = dictionary.copy()
    
    for k in range (len(keys)):
        key = keys[k]
        if (key in dict_foo.keys() ):
            dict_foo = dict_foo[key]
        else:
            dict_foo[key] = None
    dict_foo[key] = value
    return dictionary

def add_single_key_value_pair(dictionary: dict, keys: list, value):
    keys_copy = copy.deepcopy(keys)
    dictionary = append_single_key_value_pair(dictionary, keys, value)
    return dictionary

def change_single_key_value_pair(dictionary: dict, keys: list, value):
    dict_foo = dictionary
    for key in keys:
        if (key in dict_foo.keys() ):
            curr_dict = dict_foo[key]
            if len(keys) != 1:
                keys.remove(key)
                change_single_key_value_pair(curr_dict, keys, value)
            else:
                with suppress(KeyError):
                    dictionary[key] = value
        else:
            raise Exception(f"key {key} not in dictionary {dictionary}! Please check your keys list!")
    return dictionary

def overwrite_single_key_value_pair(dictionary: dict, keys: list, value):
    keys_copy = copy.deepcopy(keys)
    dictionary = change_single_key_value_pair(dictionary, keys, value)
    return dictionary

def edit_meta_data(input_dict: dict, output_dict: dict, gen_dict: dict, edit_dict: dict, read_dirpath: list, write_dirpath: list, files: list) -> str:
    list_to_check = ["add", "remove", "overwrite"]
    check_keys_in_dicts((edit_dict, ), list_to_check)

    src_file_path, file_path = get_meta_data_paths(read_dirpath, write_dirpath, files)
    src_parent_folder = os.path.split(os.path.dirname(src_file_path))[1]
    data = read.read_json(src_file_path)

    for method_key in edit_dict.keys():
        for number_key in edit_dict[method_key].keys():
            if (method_key == "remove"):
                key_list = edit_dict[method_key][number_key]["key"]
                if "type" in edit_dict[method_key]:
                    type_value = edit_dict[method_key][number_key]["type"]
                    if(type_value == "multiple ") or (type_value == "Multiple"):
                        data = remove_multiple_key_value_pairs(dictionary=copy.deepcopy(data), keys=copy.deepcopy(key_list) )
                    else:
                        raise Exception(f"Invalid type key entered in edit_dict for method_key={method_key} and number_key={number_key}!")
                else:
                    data = remove_single_key_value_pair(dictionary=copy.deepcopy(data), keys=copy.deepcopy(key_list) )
            elif(method_key == "add"):
                key_list = edit_dict[method_key][number_key]["key"]
                value = edit_dict[method_key][number_key]["value"]
                if("ref_file_to_edit" in gen_dict) and (gen_dict["ref_file_to_edit"]) and (isinstance(value, list)):
                    for i in range(len(value)):
                        if (src_parent_folder in value[i][0]):
                            idx = i  
                    value = value[idx][1]
                data = add_single_key_value_pair(dictionary = copy.deepcopy(data), keys=key_list, value=value)
            elif(method_key == "overwrite"):
                key_list = edit_dict[method_key][number_key]["key"]
                value = edit_dict[method_key][number_key]["value"]
                if("ref_file_to_edit" in gen_dict) and (gen_dict["ref_file_to_edit"]) and (isinstance(value, list)):
                    for i in range(len(value)):
                        if (src_parent_folder in value[i][0]):
                            idx = i  
                    value = value[idx][1]
                data = overwrite_single_key_value_pair(dictionary = copy.deepcopy(data), keys=copy.deepcopy(key_list), value=value)
            else:
                raise Exception("No method key was added to edit_dict dictionary! Please check edit_dict again!")
    write.write_json(file_path, data)
    return file_path

def build_edit_dict(edit_dict: dict = None, method: str = None, key: list = None, value: dict = None) -> dict:
    if method not in ["add", "remove", "overwrite"]:
        raise Exception("Invalid method entered! Please check your method choice and try again.")
    if edit_dict is None:
        edit_dict = {}
    if method not in edit_dict.keys(): # "add" or "remove" not in edit_dict:
        edit_dict[method] = {}
        idx = 1
    else:
        idx = (len(edit_dict[method].keys())) + 1
    edit_dict[method][str(idx)] = {}
    edit_dict[method][str(idx)]["key"] = key
    if method in ["add", "overwrite"]:
        edit_dict[method][str(idx)]["value"] = value
    return edit_dict