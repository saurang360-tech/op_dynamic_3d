"""
Statistical Analysis Module for Opinion Data

Provides basic statistical functions for analyzing agent opinions in 2D.
All methods expect data as (N, 2) array of [topic1, topic2] opinions.
"""

import numpy as np


def compute_stats(data):
    """
    Compute basic statistics for opinion data.
    
    Parameters:
        data: (N, 2) array of final opinions
    
    Returns:
        dict: mean, std, min, max for both topics
    """
    data = np.asarray(data)
    
    stats = {
        'topic1': {
            'mean': np.mean(data[:, 0]),
            'std': np.std(data[:, 0]),
            'min': np.min(data[:, 0]),
            'max': np.max(data[:, 0])
        },
        'topic2': {
            'mean': np.mean(data[:, 1]),
            'std': np.std(data[:, 1]),
            'min': np.min(data[:, 1]),
            'max': np.max(data[:, 1])
        }
    }
    return stats


def get_outliers(data, sigma=2.0):
    """
    Find agents beyond sigma standard deviations.
    
    Parameters:
        data: (N, 2) array of final opinions
        sigma: number of standard deviations (default: 2.0)
    
    Returns:
        dict: indices of outliers for each topic and quadrant
    """
    data = np.asarray(data)
    
    # Calculate bounds
    mean1 = np.mean(data[:, 0])
    std1 = np.std(data[:, 0])
    mean2 = np.mean(data[:, 1])
    std2 = np.std(data[:, 1])
    
    # Outlier masks
    t1_pos = data[:, 0] > mean1 + sigma * std1
    t1_neg = data[:, 0] < mean1 - sigma * std1
    t2_pos = data[:, 1] > mean2 + sigma * std2
    t2_neg = data[:, 1] < mean2 - sigma * std2
    
    # Quadrants
    q1 = t1_pos & t2_pos   # (+,+)
    q2 = t1_neg & t2_pos   # (-,+)
    q3 = t1_neg & t2_neg   # (-,-)
    q4 = t1_pos & t2_neg   # (+,-)
    
    return {
        'topic1_positive': np.where(t1_pos)[0],
        'topic1_negative': np.where(t1_neg)[0],
        'topic2_positive': np.where(t2_pos)[0],
        'topic2_negative': np.where(t2_neg)[0],
        'quadrant1': np.where(q1)[0],
        'quadrant2': np.where(q2)[0],
        'quadrant3': np.where(q3)[0],
        'quadrant4': np.where(q4)[0]
    }


def quadrant_counts(data):
    """
    Count agents in each quadrant.
    
    Parameters:
        data: (N, 2) array of final opinions
    
    Returns:
        dict: counts and percentages for each quadrant
    """
    data = np.asarray(data)
    N = len(data)
    
    q1 = np.sum((data[:, 0] > 0) & (data[:, 1] > 0))
    q2 = np.sum((data[:, 0] < 0) & (data[:, 1] > 0))
    q3 = np.sum((data[:, 0] < 0) & (data[:, 1] < 0))
    q4 = np.sum((data[:, 0] > 0) & (data[:, 1] < 0))
    
    return {
        'counts': {'Q1 (+,+)': q1, 'Q2 (-,+)': q2, 'Q3 (-,-)': q3, 'Q4 (+,-)': q4},
        'percentages': {
            'Q1 (+,+)': q1 / N * 100,
            'Q2 (-,+)': q2 / N * 100,
            'Q3 (-,-)': q3 / N * 100,
            'Q4 (+,-)': q4 / N * 100
        }
    }


def filter_outliers(data, sigma=2.0):
    """
    Remove agents outside sigma standard deviations.
    
    Parameters:
        data: (N, 2) array of final opinions
        sigma: number of standard deviations (default: 2.0)
    
    Returns:
        filtered_data: array with outliers removed
    """
    data = np.asarray(data)
    
    # Ensure it's 2D
    if data.ndim == 1:
        data = data.reshape(-1, 2)
    
    # Calculate bounds
    mean1 = np.mean(data[:, 0])
    std1 = np.std(data[:, 0])
    mean2 = np.mean(data[:, 1])
    std2 = np.std(data[:, 1])
    
    # Keep points within sigma bounds for BOTH topics
    mask = (data[:, 0] >= mean1 - sigma * std1) & \
           (data[:, 0] <= mean1 + sigma * std1) & \
           (data[:, 1] >= mean2 - sigma * std2) & \
           (data[:, 1] <= mean2 + sigma * std2)
    
    return data[mask]


def outlier_summary(data, sigma=2.0):
    """
    Print a summary of outliers and quadrant distribution.
    
    Parameters:
        data: (N, 2) array of final opinions
        sigma: number of standard deviations (default: 2.0)
    """
    data = np.asarray(data)
    N = len(data)
    
    stats = compute_stats(data)
    outliers = get_outliers(data, sigma)
    quads = quadrant_counts(data)
    
    print("=" * 50)
    print("STATISTICAL SUMMARY")
    print("=" * 50)
    print(f"Total agents: {N}")
    print()
    print("Topic 1:")
    print(f"  Mean: {stats['topic1']['mean']:.3f}")
    print(f"  Std:  {stats['topic1']['std']:.3f}")
    print(f"  Min:  {stats['topic1']['min']:.3f}")
    print(f"  Max:  {stats['topic1']['max']:.3f}")
    print()
    print("Topic 2:")
    print(f"  Mean: {stats['topic2']['mean']:.3f}")
    print(f"  Std:  {stats['topic2']['std']:.3f}")
    print(f"  Min:  {stats['topic2']['min']:.3f}")
    print(f"  Max:  {stats['topic2']['max']:.3f}")
    print()
    print(f"Outliers (> {sigma}σ):")
    print(f"  Topic 1 positive: {len(outliers['topic1_positive'])} ({len(outliers['topic1_positive'])/N*100:.1f}%)")
    print(f"  Topic 1 negative: {len(outliers['topic1_negative'])} ({len(outliers['topic1_negative'])/N*100:.1f}%)")
    print(f"  Topic 2 positive: {len(outliers['topic2_positive'])} ({len(outliers['topic2_positive'])/N*100:.1f}%)")
    print(f"  Topic 2 negative: {len(outliers['topic2_negative'])} ({len(outliers['topic2_negative'])/N*100:.1f}%)")
    print()
    print("Quadrant distribution:")
    for key, count in quads['counts'].items():
        print(f"  {key}: {count} ({quads['percentages'][key]:.1f}%)")
    print("=" * 50)