#!/bin/python3
import numpy as np
import pandas as pd
from minepy import MINE
from multiprocessing import Pool, cpu_count
from itertools import combinations
import time
import sys

def calculate_mic_pair(args):
    """
    Calculate MIC for a single pair of variables.
    
    Parameters:
    -----------
    args : tuple
        (i, j, data_i, data_j, alpha, c)
    
    Returns:
    --------
    tuple : (i, j, mic_value)
    """
    i, j, data_i, data_j, alpha, c = args
    
    mine = MINE(alpha=alpha, c=c)
    mine.compute_score(data_i, data_j)
    
    return (i, j, mine.mic())

def pairwise_mic(data, n_jobs=None, alpha=0.6, c=15, show_progress=True):
    """
    Calculate pairwise MIC values for all variable pairs in parallel.
    
    Parameters:
    -----------
    data : numpy.ndarray or pandas.DataFrame
        Data matrix where rows are observations and columns are variables
    n_jobs : int, optional
        Number of CPU cores to use. If None, uses all available cores.
    alpha : float, default=0.6
        Alpha parameter for MINE
    c : int, default=15
        c parameter for MINE
    show_progress : bool, default=True
        Whether to show progress bar
    
    Returns:
    --------
    pandas.DataFrame : Symmetric matrix of MIC values
    """
    # Convert to numpy array if DataFrame
    if isinstance(data, pd.DataFrame):
        columns = data.columns
        data = data.values
    else:
        columns = [f"Var_{i}" for i in range(data.shape[1])]
    
    n_vars = data.shape[1]
    
    # Determine number of jobs
    if n_jobs is None:
        n_jobs = cpu_count()
    
    print(f"Computing pairwise MIC for {n_vars} variables using {n_jobs} CPUs...")
    
    # Generate all pairs
    pairs = list(combinations(range(n_vars), 2))
    
    # Prepare arguments for parallel processing
    args_list = [(i, j, data[:, i], data[:, j], alpha, c) for i, j in pairs]
    
    # Parallel computation
    start_time = time.time()
    
    if show_progress:
        with Pool(n_jobs) as pool:
            results = []
            total = len(args_list)
            
            # Use imap to get results as they complete
            for idx, result in enumerate(pool.imap(calculate_mic_pair, args_list), 1):
                results.append(result)
                
                # Update progress
                percent = (idx / total) * 100
                filled = int(50 * idx / total)
                bar = '█' * filled + '-' * (50 - filled)
                
                sys.stdout.write(f'\r|{bar}| {percent:.1f}% ({idx}/{total})')
                sys.stdout.flush()
            
            print()  # New line after progress bar
    else:
        with Pool(n_jobs) as pool:
            results = pool.map(calculate_mic_pair, args_list)
    
    elapsed_time = time.time() - start_time
    print(f"Computation completed in {elapsed_time:.2f} seconds")
    
    # Create symmetric matrix
    mic_matrix = np.eye(n_vars)  # Diagonal is 1 (perfect self-correlation)
    
    for i, j, mic_value in results:
        mic_matrix[i, j] = mic_value
        mic_matrix[j, i] = mic_value  # Symmetric
    
    # Convert to DataFrame
    mic_df = pd.DataFrame(mic_matrix, index=columns, columns=columns)
    
    return mic_df

def main():
    """Example usage"""
    # Generate example data
    np.random.seed(42)
    X = pd.read_csv('./data/lifeline_univ_cox_sig_expression.csv', index_col=0)


    
    # Calculate pairwise MIC
    mic_results = pairwise_mic(X, n_jobs=8, alpha=0.4, c=5)  # Use 4 CPUs
    
    print("\nMIC Matrix:")
    print(mic_results.round(3))
    
    # Find strongest associations (excluding diagonal)
    mic_flat = mic_results.values.copy()
    np.fill_diagonal(mic_flat, 0)
    
    print("\nTop 5 strongest associations:")
    for _ in range(5):
        i, j = np.unravel_index(mic_flat.argmax(), mic_flat.shape)
        print(f"{mic_results.index[i]} - {mic_results.columns[j]}: {mic_flat[i, j]:.4f}")
        mic_flat[i, j] = 0
        mic_flat[j, i] = 0
    
    #save results
    mic_results.to_csv('./data/mic/pairwise_results_univ_cox_sig.csv')

if __name__ == "__main__":
    main()

