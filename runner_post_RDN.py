"""
Runner: Post-RDN Simulations

Reads parameters from pre-RDN HDF5 files, applies RDN,
and saves results to sim_data_post_RDN/.

No system parameters are hardcoded — everything comes from the file.

Usage:
    python runner_post_RDN.py
"""

import numpy as np
import h5py
import os
from ODE_solver import social_network


# ============================================================
# RDN CONFIGURATION (only RDN-specific settings)
# ============================================================

INPUT_FOLDER = "sim_data_pre_RDN"
OUTPUT_FOLDER = "sim_data_post_RDN"

# RDN settings
RDN_SELECT = 1        # 0=off, 1=simple, 2=2D, 3=hybrid
RDN_TOPIC = 0         # 0=topic1, 1=topic2, 2=both
RDN_D = 3.0           # RDN strength
RDN_SAMPLE_SIZE = 20  # RDN sample size

# Time settings for the RDN phase
DT = 0.01
T_DUR = 2.0


# ============================================================
# LOADING
# ============================================================

def load_pre_rdn_file(filepath):
    """
    Load an HDF5 file from the pre-RDN run.
    
    Returns:
        final_opinions: (N, 2) array — starting point for RDN
        params: dict of parameters from the file
    """
    with h5py.File(filepath, 'r') as f:
        final_opinions = f['final_opinions'][:]
        params = dict(f.attrs)
    return final_opinions, params


# ============================================================
# SIMULATION
# ============================================================

def run_rdn_simulation(final_opinions, params, dt, t_dur, D, sample_size,
                       rdn_select=1, rdn_topic=0):
    """
    Run RDN simulation starting from the given opinions.
    All system parameters come from `params` (read from the file).
    All RDN parameters come from the config block.
    """
    # Set seed for reproducibility
    seed = int(params.get('seed', 4242))
    np.random.seed(seed)
    
    # Create network using file parameters AND RDN config
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
        RDN_select=rdn_select,     # 0=off, 1=simple, 2=2D, 3=hybrid
        RDN_topic=rdn_topic,       # 0=topic1, 1=topic2, 2=both
        RDN_D=D,                   # RDN strength
        RDN_n=sample_size          # RDN sample size
    )
    
    # Override initial opinions with the pre-RDN final state
    sn._opinions = final_opinions.astype(np.float32)
    
    # Set up
    sn.set_activity()
    sn.set_social_interactions()
    
    # Run with RDN (RK4 for stability)
    sn.run_network_rk4(dt=dt, t_dur=t_dur)

    print(f"   RDN config: select={sn.RDN_select}, topic={sn.RDN_topic}, D={sn.RDN_D}, n={sn.RDN_n}")
    print(f"   Initial opinions std: {np.std(sn._opinions[:, 0]):.3f}")
    
    return sn.result.y, sn.result.t, sn.get_opinions()


# ============================================================
# SAVING
# ============================================================

def save_rdn_output(filepath, history, time_points, final_opinions, params, 
                    D, sample_size, rdn_select, rdn_topic):
    """
    Save RDN results to HDF5.
    """
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('opinions', data=history, compression='gzip')
        f.create_dataset('time', data=time_points)
        f.create_dataset('final_opinions', data=final_opinions, compression='gzip')
        f.create_dataset('last_history_step', data=history[:, :, -1], compression='gzip')
        
        # Preserve original parameters
        for key, value in params.items():
            f.attrs[key] = value
        
        # Add RDN metadata
        f.attrs['rdn_applied'] = True
        f.attrs['rdn_select'] = rdn_select
        f.attrs['rdn_topic'] = rdn_topic
        f.attrs['rdn_D'] = D
        f.attrs['rdn_sample_size'] = sample_size


# ============================================================
# BATCH RUNNER
# ============================================================

def run_batch():
    """
    Process every .h5 file in INPUT_FOLDER and save RDN results to OUTPUT_FOLDER.
    """
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Get all files from the pre-RDN folder
    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".h5")]
    files.sort()
    
    if not files:
        print(f"No .h5 files found in {INPUT_FOLDER}")
        return
    
    print("=" * 60)
    print("POST-RDN RUNNER")
    print(f"   Input:  {INPUT_FOLDER}/  ({len(files)} files)")
    print(f"   Output: {OUTPUT_FOLDER}/")
    print(f"   RDN select: {RDN_SELECT} (0=off, 1=simple, 2=2D, 3=hybrid)")
    print(f"   RDN topic:  {RDN_TOPIC} (0=topic1, 1=topic2, 2=both)")
    print(f"   RDN D: {RDN_D}")
    print(f"   RDN sample size: {RDN_SAMPLE_SIZE}")
    print(f"   dt: {DT}, t_dur: {T_DUR}")
    print("=" * 60)
    
    
    for i, filename in enumerate(files):
        run_number = i + 1
        input_path = os.path.join(INPUT_FOLDER, filename)
        
        print(f"\nRun {run_number}/{len(files)}: {filename}")
        
        # Load pre-RDN final opinions and parameters
        final_opinions, params = load_pre_rdn_file(input_path)
        
        print(f"   Loaded: N={params.get('N')}, seed={params.get('seed')}")
        
        # Run RDN simulation
        history, time_points, new_final = run_rdn_simulation(
            final_opinions=final_opinions,
            params=params,
            dt=DT,
            t_dur=T_DUR,
            D=RDN_D,
            sample_size=RDN_SAMPLE_SIZE,
            rdn_select=RDN_SELECT,
            rdn_topic=RDN_TOPIC
        )
        
        # Save output
        output_filename = f"RDN_{filename}"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        save_rdn_output(
            filepath=output_path,
            history=history,
            time_points=time_points,
            final_opinions=new_final,
            params=params,
            D=RDN_D,
            sample_size=RDN_SAMPLE_SIZE,
            rdn_select=RDN_SELECT,
            rdn_topic=RDN_TOPIC
        )
        
        print(f"   Saved: {output_filename}")
    
    print("\n" + "=" * 60)
    print(f"All {len(files)} files processed")
    print(f"   Saved in: {OUTPUT_FOLDER}/")
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_batch()