
import numpy as np

def audrc(ground_truth, predictions, certainty):
    """
    Computes the Area Under the Directional Reliability Curve (AUDRC).
    
    Parameters:
    - ground_truth: List or numpy array of true causal directions (binary: 0 or 1)
    - predictions: List or numpy array of predicted causal directions (binary: 0 or 1)
    - certainty: List or numpy array of certainty values (higher means more certain)
    
    Returns:
    - AUDRC value (between 0 and 1)
    """
    M = len(ground_truth)
    assert M == len(predictions) == len(certainty), "All inputs must have the same length."
    
    # Sort indices by descending certainty
    sorted_indices = np.argsort(-np.array(certainty))
    
    # Reorder predictions and ground truth based on sorted certainty
    sorted_predictions = np.array(predictions)[sorted_indices]
    sorted_truth = np.array(ground_truth)[sorted_indices]
    
    # Compute AUDRC
    cumulative_accuracy = np.cumsum(sorted_predictions == sorted_truth) / np.arange(1, M + 1)
    AUDRC = np.mean(cumulative_accuracy)
    
    return AUDRC
