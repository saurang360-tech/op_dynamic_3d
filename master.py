"""
Master Script (One-Variable-at-a-Time Sweep)

Reads config.json, sweeps ONE variable at a time through its array of values,
and saves results with this folder structure:

    sweeps/
    └── <variable>_range/
        ├── <variable>_<value>/
        │   ├── sim_data_pre_RDN/
        │   │   └── pre_seed_XXXX.h5 (one per repetition)
        │   └── sim_data_post_RDN/
        │       └── post_seed_XXXX.h5 (one per repetition)
        └── ...

Fixed parameters (in `fixed_params`) stay constant across every sweep.
Range parameters (in `range_params`) are swept one variable at a time.

Usage:
    python master.py
"""

import json
import os
import time
from datetime import datetime

import numpy as np
import h5py

from ODE_solver import social_network


# ============================================================
# CONFIGURATION
# ============================================================

CONFIG_FILE = "config.json"
LOG_FILE = "sweep.log"

PRE_SUBFOLDER = "sim_data_pre_RDN"
POST_SUBFOLDER = "sim_data_post_RDN"


# ============================================================
# LOGGING
# ============================================================

def log(message, also_print=True):
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] {message}"
    
    if also_print:
        print(line)
    
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# ============================================================
# CONFIG LOADING
# ============================================================

def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


def format_value(value):
    """
    Format a value for use in a folder name.
    - Floats become 2-decimal strings (e.g., 1.0 -> "1.00")
    - Ints stay as integers (e.g., 0 -> "0")
    - Strings just become the string itself
    """
    if isinstance(value, float):
        return f"{value:.2f}"
    elif isinstance(value, int):
        return str(value)
    else:
        return str(value)


# ============================================================
# RUNNERS
# ============================================================

def run_pre_rdn(params, dt, t_dur, seed, output_folder):
    """Run one pre-RDN simulation and save to output_folder."""
    os.makedirs(output_folder, exist_ok=True)
    
    np.random.seed(seed)
    
    sn = social_network(
        N=int(params['N']),
        m=int(params['m']),
        K=float(params['K']),
        alpha=float(params['alpha']),
        alpha2=float(params['alpha2']),
        delta=float(params['delta']),
        beta=float(params['beta']),
        r=float(params['r']),
        gamma=float(params['gamma']),
        epsilon=float(params['epsilon']),
        RDN_select=0,
        RDN_topic=0,
        RDN_D=0.0,
        RDN_n=0
    )
    
    sn.set_activity()
    sn.set_social_interactions()
    sn.run_network_rk4(dt=dt, t_dur=t_dur)
    
    filename = f"pre_seed_{seed:04d}.h5"
    filepath = os.path.join(output_folder, filename)
    
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('opinions', data=sn.result.y, compression='gzip')
        f.create_dataset('time', data=sn.result.t)
        f.create_dataset('final_opinions', data=sn.get_opinions(), compression='gzip')
        f.create_dataset('last_history_step', data=sn.result.y[:, :, -1], compression='gzip')
        
        for key, value in params.items():
            f.attrs[key] = value
        f.attrs['seed'] = seed
        f.attrs['rdn_applied'] = False


def run_post_rdn(pre_folder, params, dt, t_dur, seed,
                 rdn_select, rdn_topic, rdn_D, rdn_n,
                 output_folder):
    """Run one post-RDN simulation based on pre-RDN output."""
    os.makedirs(output_folder, exist_ok=True)
    
    pre_filename = f"pre_seed_{seed:04d}.h5"
    pre_path = os.path.join(pre_folder, pre_filename)
    
    if not os.path.exists(pre_path):
        raise FileNotFoundError(f"Pre-RDN file not found: {pre_path}")
    
    with h5py.File(pre_path, 'r') as f:
        final_opinions = f['final_opinions'][:]
        file_params = dict(f.attrs)
    
    np.random.seed(seed)
    
    sn = social_network(
        N=int(file_params['N']),
        m=int(file_params['m']),
        K=float(file_params['K']),
        alpha=float(file_params['alpha']),
        alpha2=float(file_params['alpha2']),
        delta=float(file_params['delta']),
        beta=float(file_params['beta']),
        r=float(file_params['r']),
        gamma=float(file_params['gamma']),
        epsilon=float(file_params['epsilon']),
        RDN_select=rdn_select,
        RDN_topic=rdn_topic,
        RDN_D=rdn_D,
        RDN_n=rdn_n
    )
    
    sn._opinions = final_opinions.astype(np.float32)
    sn.set_activity()
    sn.set_social_interactions()
    sn.run_network_rk4(dt=dt, t_dur=t_dur)
    
    filename = f"post_seed_{seed:04d}.h5"
    filepath = os.path.join(output_folder, filename)
    
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('opinions', data=sn.result.y, compression='gzip')
        f.create_dataset('time', data=sn.result.t)
        f.create_dataset('final_opinions', data=sn.get_opinions(), compression='gzip')
        f.create_dataset('last_history_step', data=sn.result.y[:, :, -1], compression='gzip')
        
        for key, value in file_params.items():
            f.attrs[key] = value
        f.attrs['seed'] = seed
        f.attrs['rdn_applied'] = True
        f.attrs['rdn_select'] = rdn_select
        f.attrs['rdn_topic'] = rdn_topic
        f.attrs['rdn_D'] = rdn_D
        f.attrs['rdn_n'] = rdn_n


# ============================================================
# ONE-VARIABLE SWEEP
# ============================================================

def sweep_one_variable(variable_name, values, fixed_params, pre_cfg, post_cfg,
                       repetitions, seed_start, output_root):
    """
    Sweep ONE variable through its array of values.
    
    For each value:
        - Create folder sweeps/<variable>_range/<variable>_<value>/
        - Run pre-RDN (repetitions times)
        - Run post-RDN (repetitions times)
    """
    variable_root = os.path.join(output_root, f"{variable_name}_range")
    os.makedirs(variable_root, exist_ok=True)
    
    log("")
    log("=" * 60)
    log(f"SWEEPING VARIABLE: {variable_name}")
    log(f"   Values: {values}")
    log(f"   Repetitions per value: {repetitions}")
    log(f"   Folder: {variable_root}/")
    log("=" * 60)
    
    for value_idx, value in enumerate(values):
        value_str = format_value(value)
        value_folder = os.path.join(variable_root, f"{variable_name}_{value_str}")
        pre_folder = os.path.join(value_folder, PRE_SUBFOLDER)
        post_folder = os.path.join(value_folder, POST_SUBFOLDER)
        
        os.makedirs(pre_folder, exist_ok=True)
        os.makedirs(post_folder, exist_ok=True)
        
        log("")
        log("-" * 60)
        log(f"VALUE {value_idx + 1}/{len(values)}: {variable_name} = {value}")
        log(f"   Folder: {value_folder}/")
        log("-" * 60)
        
        # Build params for this value
        run_params = {**fixed_params, variable_name: value}
        
        # Save params snapshot for this value
        params_path = os.path.join(value_folder, "params.json")
        with open(params_path, "w") as f:
            json.dump(run_params, f, indent=2)
        
        # Run repetitions
        for rep in range(repetitions):
            seed = seed_start + rep
            
            log(f"   Rep {rep + 1}/{repetitions} (seed={seed})")
            
            try:
                rep_start = time.time()
                
                run_pre_rdn(
                    params=run_params,
                    dt=pre_cfg['dt'],
                    t_dur=pre_cfg['t_dur'],
                    seed=seed,
                    output_folder=pre_folder
                )
                
                run_post_rdn(
                    pre_folder=pre_folder,
                    params=run_params,
                    dt=post_cfg['dt'],
                    t_dur=post_cfg['t_dur'],
                    seed=seed,
                    rdn_select=run_params.get('RDN_select', 0),
                    rdn_topic=run_params.get('RDN_topic', 0),
                    rdn_D=run_params.get('RDN_D', 1.0),
                    rdn_n=run_params.get('RDN_n', 30),
                    output_folder=post_folder
                )
                
                elapsed = time.time() - rep_start
                log(f"      Completed in {elapsed:.1f}s")
            
            except Exception as e:
                log(f"      ERROR: {e}")


# ============================================================
# MAIN SWEEP
# ============================================================

def run_sweep(config):
    fixed_params = config['fixed_params']
    range_params = config['range_params']
    pre_cfg = config['pre_rdn']
    post_cfg = config['post_rdn']
    repetitions = config['repetitions']
    seed_start = config['seed_start']
    output_root = config['output_root']
    
    os.makedirs(output_root, exist_ok=True)
    
    # Total runs estimate
    total_runs = sum(len(v) * repetitions for v in range_params.values()) * 2
    
    log("=" * 60)
    log("MASTER SWEEP (One Variable at a Time)")
    log(f"   Variables to sweep: {list(range_params.keys())}")
    log(f"   Repetitions per value: {repetitions}")
    log(f"   Estimated total runs: {total_runs}")
    log(f"   Output root: {output_root}/")
    log("=" * 60)
    
    sweep_start_time = time.time()
    
    # Sweep each variable in order
    for variable_name, values in range_params.items():
        sweep_one_variable(
            variable_name=variable_name,
            values=values,
            fixed_params=fixed_params,
            pre_cfg=pre_cfg,
            post_cfg=post_cfg,
            repetitions=repetitions,
            seed_start=seed_start,
            output_root=output_root
        )
    
    total_elapsed = time.time() - sweep_start_time
    log("")
    log("=" * 60)
    log(f"ALL SWEEPS COMPLETE")
    log(f"   Total time: {total_elapsed / 60:.1f} min")
    log(f"   Output: {output_root}/")
    log("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    if not os.path.exists(CONFIG_FILE):
        print(f"ERROR: {CONFIG_FILE} not found")
        exit(1)
    
    config = load_config(CONFIG_FILE)
    
    print("=" * 60)
    print("MASTER SCRIPT")
    print(f"   Config: {CONFIG_FILE}")
    print(f"   Sweep: {config['sweep_name']}")
    print(f"   Output: {config['output_root']}/")
    print("=" * 60)
    
    run_sweep(config)