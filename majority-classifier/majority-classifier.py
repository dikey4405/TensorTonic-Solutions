import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    # Write code here
    labels, counts = np.unique(y_train, return_counts=True)

    labels_majority = labels[np.argmax(counts)]

    return np.full(len(X_test), labels_majority)
    
    pass