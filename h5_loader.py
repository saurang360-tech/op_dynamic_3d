"""
HDF5 Data Loader Module

A collection of utility functions for loading and inspecting
simulation data stored in HDF5 (.h5) format.

Functions:
    - load_run(): Load full simulation data from a single file      
    - load_final_opinions(): Load the integrated endpoint (fast)
    - load_last_history_step(): Load the final stored pre-step snapshot
    - load_opinions_history(): Load full opinion history            [not implimented yet in the runner]
    - load_parameters(): Load simulation parameters            
    - list_files(): List all .h5 files in a folder                 
    - load_multiple(): Load multiple runs into combined dataset     [not implimented yet]

Examples:
    import data_loader as dl
    opinions, time, final, params = dl.load_run("sim_data_log/run_0001.h5")

"""

import h5py
import numpy as np
import os

# this method loads one run file and outputs all data in an array 
def load_run(file_name: str) -> np.ndarray:
    """Load one run file and return the opinion history."""
    file_path = "sim_data_log/" + file_name
    
    with h5py.File(file_path, 'r') as f:
        opinions = f['opinions'][:]      # (N, 2, steps)
        time = f['time'][:]              # (steps,)
        final = f['final_opinions'][:]   # integrated endpoint in updated files
        params = dict(f.attrs)           # parameters
    
    return params,opinions, time, final

# this method loads one run file and outputs only the parms
def load_parameters(file_name: str) -> np.ndarray:
    """Load one run file and return the last opinions only """
    file_path = "sim_data_log/" + file_name
    
    with h5py.File(file_path, 'r') as f:
        params = dict(f.attrs)           # parameters
    
    return params

# this method loads one run file and outputs all data in an array 
def load_final_opinions(file_name: str) -> np.ndarray:
    """Load the integrated endpoint stored as ``final_opinions``."""
    file_path = "sim_data_log/" + file_name
    
    with h5py.File(file_path, 'r') as f:
        final = f['final_opinions'][:]   # (N, 2)
        params = dict(f.attrs)           # parameters
    
    return params, final

def load_last_history_step(file_name: str) -> np.ndarray:
    """Load the last state stored in the history, supporting legacy HDF5 files."""
    file_path = "sim_data_log/" + file_name

    with h5py.File(file_path, 'r') as f:
        if 'last_history_step' in f:
            return f['last_history_step'][:]
        return f['opinions'][:, :, -1]


#this method lists all the files in the folder
def list_files (folder_name: str) -> np.ndarray:
    return os.listdir(folder_name)

    """
    Load multiple runs into a combined dataset.
    
    Parameters:
        folder_name: str, folder containing .h5 files
        data_type: str, what to load ('final', 'full', 'params')
                   - 'final': load only final opinions (fastest)
                   - 'full': load full opinion history (largest)
                   - 'params': load only parameters
    
    Returns:
        Combined numpy array, or list of dicts for params
    """
def load_multiple(folder_name: str, data_type: str = "final") -> np.ndarray:

    # Get all .h5 files in the folder
    files = list_files(folder_name)
    files.sort()
    
    # Load based on data_type
    if data_type == "final":
        all_data = []
        for file in files:
            file_path = os.path.join(folder_name, file)
            with h5py.File(file_path, 'r') as f:
                all_data.append(f['final_opinions'][:])
        return np.array(all_data)
    
    elif data_type == "last_history_step":
        all_data = []
        for file in files:
            file_path = os.path.join(folder_name, file)
            with h5py.File(file_path, 'r') as f:
                if 'last_history_step' in f:
                    all_data.append(f['last_history_step'][:])
                else:
                    all_data.append(f['opinions'][:, :, -1])
        return np.array(all_data)

    elif data_type == "full":
        all_data = []
        for file in files:
            file_path = os.path.join(folder_name, file)
            with h5py.File(file_path, 'r') as f:
                all_data.append(f['opinions'][:])
        return np.array(all_data)
    
    elif data_type == "params":
        all_params = []
        for file in files:
            file_path = os.path.join(folder_name, file)
            with h5py.File(file_path, 'r') as f:
                all_params.append(dict(f.attrs))
        return all_params
    
    else:
        print(f"Invalid data_type: {data_type}. Use 'final', 'full', or 'params'.")
        return np.array([])

