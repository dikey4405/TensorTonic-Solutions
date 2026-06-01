import numpy as np

def random_forest_vote(predictions):
    """
    Compute the majority vote from multiple tree predictions.
    """
    # Write code here
    predictions = np.array(predictions)

    result = []

    for sample_preds in predictions.T:
        labels, counts = np.unique(sample_preds, return_counts=True)
        result.append(labels[np.argmax(counts)])

    return result