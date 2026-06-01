import numpy as np

def random_forest_vote(predictions):
    """
    Compute the majority vote from multiple tree predictions.
    """
    # Write code here
    predictions = np.array(predictions)
    majority_pred = []
    for pred in predictions.T:
        labels, counts = np.unique(pred, return_counts = True)
        majority_pred.append(labels[np.argmax(counts)])
    return majority_pred