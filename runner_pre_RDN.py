"""
Runner: Pre-RDN Simulations

Runs multiple simulations WITHOUT RDN and saves each one
to the sim_data_pre_RDN/ folder as an HDF5 file.

Usage:
    python runner_pre_RDN.py
"""

import numpy as np
import h5py
import os
from ODE_solver import social_network


# ============================================================
# CONFIGURATION
# ============================================================

# Simulation parameters
PARAMS = {
    'N': 1000,
    'm': 5,
    'K': 2.0,
    'alpha': 1.5,
    'alpha2': 2.0,
    'delta': 90,
    'beta': 2.0,
    'r': 0.5,
    'gamma': 2.1,
    'epsilon': 0.01,
    'RDN_select': 0  # NO RDN
}

# Simulation settings
DT = 0.01
T_DUR = 4.0
NUM_SIMS = 5
SEED_START = 4242

# Output folder
OUTPUT_FOLDER = "sim_data_pre_RDN"


# ============================================================
# MAIN RUNNER
# ============================================================

def run_single_simulation(seed, params, dt, t_dur):
    """
    Run ONE simulation without RDN.
    
    Returns:
        history: (N, 2, steps) array
        time_points: (steps,) array
        final_opinions: (N, 2) array
    """
    # Set seed for reproducibility
    np.random.seed(seed)
    
    # Create network
    sn = social_network(
        N=params['N'],
        m=params['m'],
        K=params['K'],
        alpha=params['alpha'],
        alpha2=params['alpha2'],
        delta=params['delta'],
        beta=params['beta'],
        r=params['r'],
        gamma=params['gamma'],
        epsilon=params['epsilon'],
        RDN_select=params['RDN_select']
    )
    
    # Set up
    sn.set_activity()
    sn.set_social_interactions()
    
    # Run (Euler integration, no RDN)
    sn.run_network(dt=dt, t_dur=t_dur)
    
    return sn.result.y, sn.result.t, sn.get_opinions()


def save_simulation(filepath, history, time_points, final_opinions, params, seed):
    """
    Save a simulation to HDF5.
    """
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('opinions', data=history, compression='gzip')
        f.create_dataset('time', data=time_points)
        f.create_dataset('final_opinions', data=final_opinions, compression='gzip')
        f.create_dataset('last_history_step', data=history[:, :, -1], compression='gzip')
        
        # Save parameters as attributes
        for key, value in params.items():
            f.attrs[key] = value
        f.attrs['seed'] = seed
        f.attrs['rdn_applied'] = False


def run_batch():
    """
    Run all simulations and save each one.
    """
    # Create output folder
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    print("=" * 60)
    print(f"PRE-RDN RUNNER")
    print(f"   Simulations: {NUM_SIMS}")
    print(f"   Agents: {PARAMS['N']}")
    print(f"   Seed start: {SEED_START}")
    print(f"   Output: {OUTPUT_FOLDER}/")
    print("=" * 60)
    
    for i in range(NUM_SIMS):
        current_seed = SEED_START + i
        run_number = i + 1
        
        print(f"\nRun {run_number}/{NUM_SIMS} (seed={current_seed})")
        
        # Run simulation
        history, time_points, final_opinions = run_single_simulation(
            seed=current_seed,
            params=PARAMS,
            dt=DT,
            t_dur=T_DUR
        )
        
        # Save
        filename = f"run_{run_number:04d}_seed_{current_seed:04d}.h5"
        filepath = os.path.join(OUTPUT_FOLDER, filename)
        
        save_simulation(
            filepath=filepath,
            history=history,
            time_points=time_points,
            final_opinions=final_opinions,
            params=PARAMS,
            seed=current_seed
        )
        
        print(f"   Saved: {filename}")
    
    print("\n" + "=" * 60)
    print(f"All {NUM_SIMS} simulations complete")
    print(f"   Saved in: {OUTPUT_FOLDER}/")
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_batch()