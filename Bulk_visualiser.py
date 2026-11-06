"""
Bulk Visualiser

Visualizes pre-RDN vs post-RDN from a value folder inside a sweep.

Usage:
    # Visualize one specific value folder
    python Bulk_visualiser.py delta_range delta_30

    # Visualize one value folder in a custom root
    python Bulk_visualiser.py alpha_range alpha_1.00 --root sweeps

    # Loop through all values in a sweep
    python Bulk_visualiser.py delta_range

    # List all available folders (no plot)
    python Bulk_visualiser.py --list
    python Bulk_visualiser.py delta_range --list

    # Custom root
    python Bulk_visualiser.py delta_range delta_30 --root my_sweeps
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

import stats


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_ROOT = "sweeps"
PRE_SUBFOLDER = "sim_data_pre_RDN"
POST_SUBFOLDER = "sim_data_post_RDN"

SIGMA = 3.0
BUFFER = 0.10

BINS = 100
CMAP = "hot"
DOT_SIZE = 2.0
DOT_ALPHA = 0.8

USE_LOG = True
POWER = 0.7
PERCENTILE = 95


# ============================================================
# LOADING
# ============================================================

def load_folder(folder):
    """
    Load all .h5 files in a folder and combine their integrated endpoints.

    Newly written files use final_opinions for the post-final-step state;
    legacy files remain readable with their original stored values.
    """
    if not os.path.isdir(folder):
        print(f"   Folder not found: {folder}")
        return None, 0
    
    files = [f for f in os.listdir(folder) if f.endswith(".h5")]
    files.sort()
    
    if not files:
        print(f"   No .h5 files in {folder}")
        return None, 0
    
    runs = []
    for f in files:
        path = os.path.join(folder, f)
        with h5py.File(path, 'r') as hf:
            runs.append(hf['final_opinions'][:])
    
    combined = np.vstack(runs)
    print(f"   Loaded {len(files)} files, {len(combined)} agents from {folder}")
    return combined, len(files)


# ============================================================
# FILTERING
# ============================================================

def filter_data(data, sigma=2.0):
    return stats.filter_outliers(data, sigma=sigma)


# ============================================================
# AUTO-SCALE
# ============================================================

def get_limit(data, buffer=BUFFER):
    max_abs = np.max(np.abs(data))
    return max_abs * (1 + buffer)


# ============================================================
# DENSITY LOOKUP
# ============================================================

def compute_density_per_agent(data, bins, limit):
    hist, x_edges, y_edges = np.histogram2d(
        data[:, 0], data[:, 1],
        bins=bins,
        range=[[-limit, limit], [-limit, limit]]
    )
    x_idx = np.clip(np.digitize(data[:, 0], x_edges) - 1, 0, bins - 1)
    y_idx = np.clip(np.digitize(data[:, 1], y_edges) - 1, 0, bins - 1)
    return hist[x_idx, y_idx]


def transform_density(density, use_log=True, power=0.7):
    d = density.astype(np.float64)
    if use_log:
        d = np.log1p(d)
    d = d ** power
    return d


# ============================================================
# VISUALISATION
# ============================================================

def plot_comparison(pre_data, post_data, title):
    pre_filtered = filter_data(pre_data, SIGMA)
    post_filtered = filter_data(post_data, SIGMA)
    
    pre_limit = get_limit(pre_filtered)
    post_limit = get_limit(post_filtered)
    
    print(f"   Pre-RDN axis limit:  ±{pre_limit:.3f}")
    print(f"   Post-RDN axis limit: ±{post_limit:.3f}")
    
    pre_density = compute_density_per_agent(pre_filtered, BINS, pre_limit)
    post_density = compute_density_per_agent(post_filtered, BINS, post_limit)
    
    pre_density = transform_density(pre_density, USE_LOG, POWER)
    post_density = transform_density(post_density, USE_LOG, POWER)
    
    vmax = max(
        np.percentile(pre_density, PERCENTILE),
        np.percentile(post_density, PERCENTILE)
    )
    
    print(f"   Density vmax (p{PERCENTILE}): {vmax:.3f}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    for ax in (ax1, ax2):
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('white')
    
    sc1 = ax1.scatter(
        pre_filtered[:, 0], pre_filtered[:, 1],
        c=pre_density, s=DOT_SIZE, cmap=CMAP,
        alpha=DOT_ALPHA, vmin=0, vmax=vmax, edgecolors='none'
    )
    ax1.axhline(y=0, color='cyan', linewidth=1, alpha=0.5)
    ax1.axvline(x=0, color='cyan', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Topic 1 Opinion', fontsize=12)
    ax1.set_ylabel('Topic 2 Opinion', fontsize=12)
    ax1.set_title(f'Pre-RDN ({len(pre_filtered)} agents)', fontsize=13)
    ax1.set_xlim(-pre_limit, pre_limit)
    ax1.set_ylim(-pre_limit, pre_limit)
    ax1.set_aspect('equal')
    plt.colorbar(sc1, ax=ax1, fraction=0.046, label='Density')
    
    sc2 = ax2.scatter(
        post_filtered[:, 0], post_filtered[:, 1],
        c=post_density, s=DOT_SIZE, cmap=CMAP,
        alpha=DOT_ALPHA, vmin=0, vmax=vmax, edgecolors='none'
    )
    ax2.axhline(y=0, color='cyan', linewidth=1, alpha=0.5)
    ax2.axvline(x=0, color='cyan', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Topic 1 Opinion', fontsize=12)
    ax2.set_ylabel('Topic 2 Opinion', fontsize=12)
    ax2.set_title(f'Post-RDN ({len(post_filtered)} agents)', fontsize=13)
    ax2.set_xlim(-post_limit, post_limit)
    ax2.set_ylim(-post_limit, post_limit)
    ax2.set_aspect('equal')
    plt.colorbar(sc2, ax=ax2, fraction=0.046, label='Density')
    
    fig.patch.set_facecolor('black')
    plt.suptitle(title, fontsize=16, color='white')
    plt.tight_layout()
    plt.show()


# ============================================================
# STATISTICS
# ============================================================

def print_statistics(pre_data, post_data):
    print("\n" + "=" * 60)
    print("STATISTICS COMPARISON")
    print("=" * 60)
    
    for label, data in [("Pre-RDN", pre_data), ("Post-RDN", post_data)]:
        s = stats.compute_stats(data)
        print(f"\n{label}:")
        print(f"   Agents: {len(data)}")
        print(f"   Topic 1: mean={s['topic1']['mean']:.3f}, "
              f"std={s['topic1']['std']:.3f}, "
              f"range=[{s['topic1']['min']:.2f}, {s['topic1']['max']:.2f}]")
        print(f"   Topic 2: mean={s['topic2']['mean']:.3f}, "
              f"std={s['topic2']['std']:.3f}, "
              f"range=[{s['topic2']['min']:.2f}, {s['topic2']['max']:.2f}]")
    
    print("\n" + "-" * 60)
    print("CHANGE (Post - Pre):")
    print(f"   Topic 1 std change: {np.std(post_data[:, 0]) - np.std(pre_data[:, 0]):.3f}")
    print(f"   Topic 2 std change: {np.std(post_data[:, 1]) - np.std(pre_data[:, 1]):.3f}")
    print("=" * 60)


# ============================================================
# VISUALISE ONE VALUE
# ============================================================

def visualise_value(root, sweep_name, value_name):
    """
    Visualize one value folder.
    """
    value_folder = os.path.join(root, sweep_name, value_name)
    pre_folder = os.path.join(value_folder, PRE_SUBFOLDER)
    post_folder = os.path.join(value_folder, POST_SUBFOLDER)
    
    if not os.path.isdir(value_folder):
        print(f"ERROR: Folder not found: {value_folder}")
        return
    
    title = f"{sweep_name} / {value_name}"
    
    print("=" * 60)
    print(f"VISUALISING: {title}")
    print("=" * 60)
    
    print(f"\nLoading pre-RDN from {pre_folder}/")
    pre_data, _ = load_folder(pre_folder)
    
    print(f"\nLoading post-RDN from {post_folder}/")
    post_data, _ = load_folder(post_folder)
    
    if pre_data is None or post_data is None:
        print("\nERROR: Could not load matching data")
        return
    
    print_statistics(pre_data, post_data)
    
    print("\nGenerating comparison plot...")
    plot_comparison(pre_data, post_data, title)
    
    print("\nDone.")


# ============================================================
# LIST HELPERS
# ============================================================

def list_sweeps(root):
    """List all sweep folders in the root."""
    if not os.path.isdir(root):
        print(f"Root folder not found: {root}")
        return []
    
    sweeps = []
    for entry in sorted(os.listdir(root)):
        path = os.path.join(root, entry)
        if os.path.isdir(path):
            sweeps.append(entry)
    return sweeps


def list_values(root, sweep_name):
    """List all value folders inside a sweep."""
    sweep_folder = os.path.join(root, sweep_name)
    
    if not os.path.isdir(sweep_folder):
        print(f"ERROR: Sweep folder not found: {sweep_folder}")
        return []
    
    values = []
    for entry in sorted(os.listdir(sweep_folder)):
        path = os.path.join(sweep_folder, entry)
        # Only count as a value folder if it has the pre sub-folder
        if os.path.isdir(path) and os.path.isdir(os.path.join(path, PRE_SUBFOLDER)):
            values.append(entry)
    return values


def print_available(root, sweep_name=None):
    """
    Print the folder structure available for visualization.
    """
    print("=" * 60)
    print("AVAILABLE FOLDERS")
    print("=" * 60)
    print(f"Root: {root}/")
    
    sweeps = list_sweeps(root)
    
    if not sweeps:
        print("   No sweeps found.")
        return
    
    if sweep_name is not None:
        # Show values for one sweep
        if sweep_name not in sweeps:
            print(f"   Sweep '{sweep_name}' not found.")
            return
        values = list_values(root, sweep_name)
        print(f"\n{sweep_name}/")
        for v in values:
            print(f"   {sweep_name}/{v}/")
    else:
        # Show all sweeps and values
        for s in sweeps:
            values = list_values(root, s)
            print(f"\n{s}/")
            for v in values:
                print(f"   {s}/{v}/")
    
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize pre-RDN vs post-RDN from a value folder."
    )
    parser.add_argument('sweep_name', nargs='?', default=None,
                        help='Sweep folder name (e.g., delta_range)')
    parser.add_argument('value_name', nargs='?', default=None,
                        help='Value folder name (e.g., delta_30). '
                             'If omitted, loops through all values.')
    parser.add_argument('--root', default=DEFAULT_ROOT,
                        help=f'Root folder (default: {DEFAULT_ROOT})')
    parser.add_argument('--list', action='store_true',
                        help='List available folders and exit (no plot)')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Handle --list
    if args.list:
        print_available(args.root, args.sweep_name)
        return
    
    # If no sweep specified at all
    if args.sweep_name is None:
        print_available(args.root)
        print("\nUsage:")
        print("   python Bulk_visualiser.py <sweep_name> <value_name>")
        print("   python Bulk_visualiser.py <sweep_name>")
        return
    
    # Visualize one specific value
    if args.value_name is not None:
        visualise_value(args.root, args.sweep_name, args.value_name)
        return
    
    # Loop through all values in the sweep
    values = list_values(args.root, args.sweep_name)
    
    if not values:
        print(f"No value folders found in {args.root}/{args.sweep_name}")
        return
    
    print("=" * 60)
    print(f"SWEEP: {args.sweep_name}")
    print(f"   Values found: {values}")
    print("=" * 60)
    
    for value_name in values:
        visualise_value(args.root, args.sweep_name, value_name)


if __name__ == "__main__":
    main()